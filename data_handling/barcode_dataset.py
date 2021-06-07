import cv2
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from collections import Counter
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from utils import Utils
from scipy.spatial.transform import Rotation as R


class BarcodeDataset:
    def __init__(self, dataset_path):
        self.datasetPath = dataset_path
        self.allImagePaths = [os.path.join(self.datasetPath, f)
                              for f in listdir(self.datasetPath) if isfile(join(self.datasetPath, f))]
        self.anchorBoxes = None
        self.boundingBoxDict = None

    def calculate_text_bounding_boxes(self, barcodes_with_text_detection_path):
        all_paths = [os.path.join(barcodes_with_text_detection_path, f)
                     for f in listdir(barcodes_with_text_detection_path)
                     if isfile(join(barcodes_with_text_detection_path, f))]
        # assert len(all_paths) % 4 == 0
        # All barcode image files and their corresponding data
        barcode_file_path_dict = {}
        barcode_data_path_dict = {}

        for path in all_paths:
            file_name = os.path.split(path)[1]
            if ".png" in file_name and "_mask" not in file_name and "_bb" not in file_name:
                barcode_file_path_dict[file_name[:-4]] = path
            if ".dat" in file_name:
                barcode_data_path_dict[file_name[:file_name.index("_")]] = path
        files_set1 = set(barcode_file_path_dict.keys())
        files_set2 = set(barcode_data_path_dict.keys())
        assert files_set1 == files_set2
        # All barcode image sizes and bounding box list for texts
        barcode_image_sizes_dict = {}
        barcode_text_bbs_dict = {}
        for file_name in tqdm(barcode_file_path_dict):
            image = cv2.imread(barcode_file_path_dict[file_name])
            data_path = barcode_data_path_dict[file_name]
            with open(data_path, "rb") as f:
                data_dict = pickle.load(f)
            # "bounding_box": np.stack([scaled_top_left, scaled_bottom_right], axis=0)
            bounding_box = data_dict["bounding_box"]
            bounding_box = np.array([
                data_dict["bounding_box"][0, 1],
                data_dict["bounding_box"][0, 0],
                data_dict["bounding_box"][1, 1],
                data_dict["bounding_box"][1, 0]
            ])
            # bounding_box_normalized = np.clip(
            #     np.array(
            #         [bounding_box[0, 1] / image.shape[1],
            #          bounding_box[0, 0] / image.shape[0],
            #          bounding_box[1, 1] / image.shape[1],
            #          bounding_box[1, 0] / image.shape[0]]), 0.0, 1.0)
            # Utils.show_image_with_normalized_bounding_boxes(image, [bounding_box_normalized], "normalized_bb")
            barcode_image_sizes_dict[file_name] = np.array([image.shape[0], image.shape[1]])
            barcode_text_bbs_dict[file_name] = bounding_box
        # Analyze image sizes
        image_sizes = np.stack([shp for shp in barcode_image_sizes_dict.values()], axis=0)
        height_width_ratios = np.array([shp[0] / shp[1] for shp in image_sizes])
        assert all([image_sizes[i, 0] / image_sizes[i, 1] == height_width_ratios[i]
                    for i in range(image_sizes.shape[0])])
        mean_size = np.mean(image_sizes, axis=0)
        print("Mean Size:{0}".format(mean_size))
        print("Min Ratio:{0}".format(np.min(height_width_ratios)))
        print("Max Ratio:{0}".format(np.max(height_width_ratios)))
        # Cluster bounding boxes
        bb_coords = np.stack([bb for bb in barcode_text_bbs_dict.values()], axis=0)
        self.boundingBoxDict = {}
        for k, v in barcode_text_bbs_dict.items():
            self.boundingBoxDict[os.path.join(barcodes_with_text_detection_path, "{0}.png".format(k))] = [v]

        # self.anchorBoxes = BarcodeDataset.cluster_bounding_boxes(bb_coords=bb_coords, cluster_count=cluster_count)
        # self.boundingBoxDict = {}

    def calculate_mask_oriented_bounding_boxes(self, barcodes_with_text_detection_path, verbose=False):
        all_paths = [os.path.join(barcodes_with_text_detection_path, f)
                     for f in listdir(barcodes_with_text_detection_path)
                     if isfile(join(barcodes_with_text_detection_path, f))]

        for path in tqdm(all_paths):
            if "_mask.png" not in path:
                continue
            file_name = os.path.split(path)[1][:-4]
            mask_image_3c = cv2.imread(path)
            assert np.array_equal(mask_image_3c[..., 0], mask_image_3c[..., 1]) and \
                   np.array_equal(mask_image_3c[..., 0], mask_image_3c[..., 2])
            mask_image = mask_image_3c[..., 0]
            white_coords = np.nonzero(mask_image)
            white_coords = np.stack([white_coords[1], white_coords[0]], axis=1).astype(np.float32)
            pca = PCA()
            pca.fit(white_coords)
            center_of_mass = np.mean(white_coords, axis=0)
            components = pca.components_.copy()
            components[0] = np.sign(components[0, 0]) * components[0]
            components[1] = np.sign(components[1, 1]) * components[1]
            coeff = 1.0
            if components[0, 1] >= 0:
                coeff = -coeff
            angle = coeff * np.rad2deg(np.arccos(np.clip(components[0, 0], -1.0, 1.0)))

            r = R.from_euler('Z', angle, degrees=True).as_matrix()
            # Translate middle point to the origin
            t0 = np.eye(N=3)
            dx = -center_of_mass[0]
            dy = -center_of_mass[1]
            t0[0, 2] = dx
            t0[1, 2] = dy
            # Translate back
            t1 = np.eye(N=3)
            t1[0, 2] = -dx
            t1[1, 2] = -dy
            # Final transformation matrix
            m = t1 @ (r @ t0)
            transformed_image = cv2.warpAffine(mask_image, m[0:2, :], (mask_image.shape[1], mask_image.shape[0]))
            # Axis aligned box
            white_coords_transformed = np.nonzero(transformed_image)
            white_coords_transformed = np.stack([white_coords_transformed[1],
                                                 white_coords_transformed[0]], axis=1).astype(np.float32)
            left_top = np.min(white_coords_transformed, axis=0)
            right_bottom = np.max(white_coords_transformed, axis=0)
            left_bottom = np.array([left_top[0], right_bottom[1]])
            right_top = np.array([right_bottom[0], left_top[1]])
            aabb_coordinates = np.stack([
                left_top,
                right_top,
                right_bottom,
                left_bottom
            ], axis=0)
            m_inv = np.linalg.inv(m)
            obb_coordinates = Utils.affine_transform_points_2d(m_inv, aabb_coordinates)

            pickle_path = os.path.join(barcodes_with_text_detection_path, "{0}_obb_coordinates.dat".format(file_name))
            with open(pickle_path, "wb") as f:
                pickle.dump(obb_coordinates, f)

            # Visualization
            if verbose:
                mask_image_3c_copy = mask_image_3c.copy()
                for j in range(obb_coordinates.shape[0]):
                    p1 = (int(obb_coordinates[j][0]), int(obb_coordinates[j][1]))
                    p2 = (int(obb_coordinates[(j + 1) % 4][0]), int(obb_coordinates[(j + 1) % 4][1]))
                    cv2.line(mask_image_3c_copy, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow("mask_image", mask_image)
                cv2.imshow("transformed_image", transformed_image)
                cv2.imshow("mask_image_obb", mask_image_3c_copy)
                cv2.waitKey(0)
                print("X")

        # file_path = img_path.numpy().decode("utf-8")
        # if file_path not in self.imageDict:
        #     assert file_path not in self.maskDict
        #     image = cv2.imread(file_path)
        #     mask_image = cv2.imread(file_path[:-4] + "_mask.png")
        #     self.imageDict[file_path] = image
        #     self.maskDict[file_path] = mask_image
        # else:
        #     assert file_path in self.maskDict
        #     image = self.imageDict[file_path]
        #     mask_image = self.maskDict[file_path]
        # # Get non zero positions
        # non_zero_indices = np.nonzero(mask_image)
        # print("X")
