import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial.transform import Rotation as R
import shutil
import pickle

from utils import Utils


class TextAnnotator:
    def __init__(self):
        pass

    @staticmethod
    def rotation_normalization(image, detection, verbose=True):
        image_copy = image.copy()
        image_grayscale = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        ret2, thresholded_image = cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thresholded_image_3c = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
        black_coords = np.nonzero(thresholded_image == 0)
        black_coords = np.stack([black_coords[0], black_coords[1]], axis=1).astype(np.float32)
        polygon = Polygon([detection[0], detection[1], detection[2], detection[3]])
        barcode_coords = []

        for j in range(4):
            p1 = (int(detection[j][0]), int(detection[j][1]))
            p2 = (int(detection[(j + 1) % 4][0]), int(detection[(j + 1) % 4][1]))
            cv2.line(image_copy, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)

        for black_coord in black_coords:
            point = Point(black_coord[1], black_coord[0])
            does_contain = polygon.contains(point)
            if does_contain:
                image_copy[black_coord.astype(np.int32)[0], black_coord.astype(np.int32)[1], :] = np.array([0, 0, 255])
                barcode_coords.append(black_coord)
        barcode_coords = np.stack(barcode_coords, axis=0)
        pca = PCA()
        pca.fit(barcode_coords)
        components = pca.components_.copy()
        components[0] = np.sign(components[0, 1]) * components[0]
        components[1] = np.sign(components[1, 0]) * components[1]
        angle = np.rad2deg(np.arccos(np.clip(components[0, 1], -1.0, 1.0)))
        if components[0, 0] >= 0:
            angle = -angle
        r = R.from_euler('Z', angle, degrees=True).as_matrix()

        if verbose:
            center_coord = 0.5 * np.array([image_copy.shape[0], image_copy.shape[1]])
            component_x = 25.0 * components[0]
            component_y = 25.0 * components[1]
            p_x = component_x + center_coord
            p_y = component_y + center_coord
            cv2.line(image_copy, (int(center_coord[1]), int(center_coord[0])), (int(p_x[1]), int(p_x[0])),
                     (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(image_copy, (int(center_coord[1]), int(center_coord[0])), (int(p_y[1]), int(p_y[0])),
                     (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("PCA Axes and Contained Points", image_copy)

            # Rotate the image
            transformed_image = cv2.warpAffine(image_copy, r[0:2, :], (image_copy.shape[1], image_copy.shape[0]))
            cv2.imshow("Rotated Image", transformed_image)

        transformed_image_original = cv2.warpAffine(image.copy(), r[0:2, :], (image_copy.shape[1], image_copy.shape[0]))
        return transformed_image_original, angle

    @staticmethod
    def connected_components_analysis(image, detection):
        image_copy = image.copy()
        image_grayscale = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        ret2, thresholded_image = cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholded_image_not = np.zeros_like(thresholded_image)
        cv2.bitwise_not(thresholded_image, thresholded_image_not)
        # cv2.imshow("thresholded_image", thresholded_image)
        # cv2.waitKey(0)

        # thresholded_image_not = np.zeros_like(thresholded_image)
        # cv2.bitwise_not(thresholded_image, thresholded_image_not)

        # results = []
        # for bin_image in [thresholded_image, thresholded_image_not]:
        #     output = cv2.connectedComponentsWithStats(bin_image, 8, cv2.CV_32S)
        #     results.append(output)
        #
        # if results[0][0] > results[1][0]:
        #     final_image = thresholded_image
        #     output = results[0]
        # else:
        #     final_image = thresholded_image_not
        #     output = results[1]

        output = cv2.connectedComponentsWithStats(thresholded_image_not, 8, cv2.CV_32S)
        polygon = Polygon([detection[0], detection[1], detection[2], detection[3]])
        (numLabels, labels, stats, centroids) = output
        selected_labels = set()
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                label_id = labels[i, j]
                if label_id == 0:
                    continue
                point = Point(j, i)
                does_contain = polygon.contains(point)
                if does_contain:
                    selected_labels.add(label_id)

        mask_arr = np.zeros_like(labels)
        for selected_label in selected_labels:
            mask_arr = np.logical_or(mask_arr, (labels == selected_label).astype(mask_arr.dtype))

        mask_arr = mask_arr.astype(np.uint8)
        mask_arr[mask_arr == 1] = 255
        cv2.imshow("Mask Array", mask_arr)
        # cv2.waitKey(0)

        # All nonzero pixels
        white_coords = np.nonzero(mask_arr == 255)
        white_coords = np.stack([white_coords[0], white_coords[1]], axis=1)
        top_left = np.min(white_coords, axis=0)
        bottom_right = np.max(white_coords, axis=0)

        show_image = thresholded_image_not.copy()
        show_image = cv2.cvtColor(show_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(show_image, (top_left[1], top_left[0]), (bottom_right[1], bottom_right[0]),
                      color=(0, 0, 255), thickness=1)

        cv2.imshow("Connected Components", show_image)
        # cv2.waitKey(0)

        bounding_box = np.stack([top_left, bottom_right], axis=0)
        return bounding_box, stats, centroids, selected_labels, mask_arr

    @staticmethod
    def annotate_files(text_detector,
                       unprocessed_files_path,
                       processed_files_path,
                       problematic_files_path,
                       bb_scale_ratio):
        unprocessed_files = [os.path.join(unprocessed_files_path, f)
                             for f in listdir(unprocessed_files_path) if isfile(join(unprocessed_files_path, f))]

        for img_path in unprocessed_files:
            file_name = os.path.split(img_path)[1][:-4]
            image = cv2.imread(img_path)
            if image is None:
                continue
            # Step 1: Detect text bounding boxes
            detection = text_detector.get_detections(image=image, filter=True)
            # Step 2: Normalize for rotation.  We use the detected text from the previous step for detecting the
            # rotation angle.
            normalized_image, angle = TextAnnotator.rotation_normalization(image=image, detection=detection,
                                                                           verbose=True)
            # Step 3: Detect text in the rotation normalized image. This detections tends to be more correct.
            detection_normalized = text_detector.get_detections(image=normalized_image, filter=True)
            Utils.show_text_detections(image=normalized_image,
                                       detections=[detection_normalized],
                                       window_name="Normalized Detections")
            # Step 4: Rotate the detection in the normalized space back into the original image coords.
            r_back = R.from_euler('Z', -angle, degrees=True).as_matrix()
            detection_normalized_in_original = Utils.affine_transform_points_2d(r_back, detection_normalized)
            original_image_copy = image.copy()
            Utils.show_text_detections(image=original_image_copy,
                                       detections=[detection_normalized_in_original],
                                       window_name="Normalized Detections in Original Image")
            # Step 5: Connected components analysis for tight fit around images. Scale the bounding box by its
            # diagonal so it also takes a certain margin around the barcode
            barcode_bounding_box, stats, centroids, selected_labels, mask_arr \
                = TextAnnotator.connected_components_analysis(image=original_image_copy,
                                                              detection=detection_normalized_in_original)
            diagonal_length = np.linalg.norm(barcode_bounding_box[0] - barcode_bounding_box[1])
            center_point = barcode_bounding_box[0] + 0.5 * (barcode_bounding_box[1] - barcode_bounding_box[0])
            diagonal_vector = (barcode_bounding_box[1] - barcode_bounding_box[0]) / diagonal_length
            diagonal_length_with_margin = bb_scale_ratio * diagonal_length
            scaled_top_left = center_point - 0.5 * diagonal_length_with_margin * diagonal_vector
            scaled_bottom_right = center_point + 0.5 * diagonal_length_with_margin * diagonal_vector
            scaled_top_left = scaled_top_left.astype(np.int32)
            scaled_bottom_right = scaled_bottom_right.astype(np.int32)
            final_image_with_bounding_box = image.copy()
            result_dict = {
                "bounding_box": np.stack([scaled_top_left, scaled_bottom_right], axis=0),
                "stats": stats,
                "centroids": centroids,
                "selected_labels": selected_labels
            }
            cv2.rectangle(final_image_with_bounding_box,
                          (result_dict["bounding_box"][0, 1], result_dict["bounding_box"][0, 0]),
                          (result_dict["bounding_box"][1, 1], result_dict["bounding_box"][1, 0]),
                          color=(0, 0, 255), thickness=1)
            cv2.imshow("Final Bounding Box", final_image_with_bounding_box)
            res = cv2.waitKey(0)
            # Space key; a good annotation; save this.
            if res == 32:
                destination_path = processed_files_path
            else:
                destination_path = problematic_files_path

            dest = shutil.move(img_path, destination_path)
            pickle_path = os.path.join(destination_path, "{0}_detection_stats.dat".format(file_name))
            cv2.imwrite(os.path.join(destination_path, "{0}_mask.png".format(file_name)), mask_arr)
            cv2.imwrite(os.path.join(destination_path, "{0}_bb.png".format(file_name)), final_image_with_bounding_box)
            with open(pickle_path, "wb") as f:
                pickle.dump(result_dict, f)
