import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import cv2
import pickle
from scipy.spatial import distance_matrix
from sklearn.mixture import GaussianMixture

from utils import Utils


class BoundingBoxGMMModeller(object):
    def __init__(self, cluster_count):
        self.clusterCount = cluster_count
        self.allIoUs = []
        self.allBbs = []
        self.allHuMoments = []
        self.allFeatureVectors = []
        self.gmm = None

    def train_gmm_model(self, train_paths, dilation_kernel_size, output_path, verbose=True):
        Utils.create_directory(path=output_path)

        self.allIoUs = []
        self.allBbs = []
        self.allHuMoments = []
        self.allFeatureVectors = []
        for path_id, path in tqdm(enumerate(train_paths)):
            root_path = os.path.split(path)[0]
            file_name = os.path.split(path)[1]
            mask_image_path = os.path.join(root_path, file_name[:-4] + "_mask.png")
            detection_stats_path = os.path.join(root_path, file_name[:-4] + "_detection_stats.dat")
            with open(detection_stats_path, "rb") as f:
                detections_dict = pickle.load(f)
            # Ground truth bounding box
            gt_bounding_box = np.array(
                [detections_dict["bounding_box"][0, 1],
                 detections_dict["bounding_box"][0, 0],
                 detections_dict["bounding_box"][1, 1],
                 detections_dict["bounding_box"][1, 0]])
            gt_bounding_box_area = (gt_bounding_box[2] - gt_bounding_box[0]) * (gt_bounding_box[3] - gt_bounding_box[1])
            mask_image = tf.io.decode_png(tf.io.read_file(mask_image_path))
            dilated_mask_image = tf.nn.dilation2d(
                input=tf.expand_dims(mask_image, axis=0),
                filters=tf.zeros(shape=[dilation_kernel_size, dilation_kernel_size, 1], dtype=tf.uint8),
                strides=[1, 1, 1, 1],
                padding="SAME",
                data_format="NHWC",
                dilations=[1, 1, 1, 1]
            )
            dilated_mask_image = tf.squeeze(tf.cast(tf.greater(dilated_mask_image, 0), dtype=tf.uint8)).numpy()
            if verbose:
                cv2.imshow("Dilated Mask", 255 * dilated_mask_image)
                cv2.waitKey(0)
            # Calculate bounding box around the silhouette of the characters.
            vertical_sum = np.sum(dilated_mask_image, axis=1)
            horizontal_sum = np.sum(dilated_mask_image, axis=0)
            top = -1
            bottom = -1
            for y_id in range(vertical_sum.shape[0]):
                if vertical_sum[y_id] > 0:
                    if top == -1:
                        top = y_id
                    bottom = y_id
            left = -1
            right = -1
            for x_id in range(horizontal_sum.shape[0]):
                if horizontal_sum[x_id] > 0:
                    if left == -1:
                        left = x_id
                    right = x_id
            cc_bounding_box = np.array([left, top, right, bottom])
            # Calculate IoU metric
            intersection_left = max(gt_bounding_box[0], cc_bounding_box[0])
            intersection_top = max(gt_bounding_box[1], cc_bounding_box[1])
            intersection_right = min(gt_bounding_box[2], cc_bounding_box[2])
            intersection_bottom = min(gt_bounding_box[3], cc_bounding_box[3])
            intersection_area = max(0, intersection_right - intersection_left) * max(
                0, intersection_bottom - intersection_top)
            cc_bounding_box_area = (cc_bounding_box[2] - cc_bounding_box[0]) * (
                    cc_bounding_box[3] - cc_bounding_box[1])
            union_area = gt_bounding_box_area + cc_bounding_box_area - intersection_area
            iou = intersection_area / union_area
            # if iou < 0.718:
            #     print("X")
            self.allIoUs.append(iou)
            # Features to calculate:
            # 1) Normalized bounding box coordinates
            width = dilated_mask_image.shape[1]
            height = dilated_mask_image.shape[0]
            cc_bounding_box_normalized = np.array([cc_bounding_box[0] / width,
                                                   cc_bounding_box[1] / height,
                                                   cc_bounding_box[2] / width,
                                                   cc_bounding_box[3] / height])
            self.allBbs.append(cc_bounding_box_normalized)
            # 2) Calculate the Hu moments of silhouette
            cropped_image = dilated_mask_image[top:bottom, left:right]
            moments = cv2.moments(cropped_image, binaryImage=True)
            hu_moments = cv2.HuMoments(moments)[:, 0]
            self.allHuMoments.append(hu_moments)
            feature_vector = np.concatenate([cc_bounding_box_normalized, hu_moments])
            self.allFeatureVectors.append(feature_vector)

        # Train GMM
        self.gmm = GaussianMixture(n_components=self.clusterCount, covariance_type="diag")
        X = np.stack(self.allFeatureVectors, axis=0)
        self.gmm.fit(X)

        with open(os.path.join(output_path, "gmm_model.sav"), "wb") as f:
            pickle.dump(self, f)

    def get_gmm_feature_from_bounding_box(self, image, bounding_box):
        # Convert to pixel coordinates
        width = image.shape[1]
        height = image.shape[0]
        # left = int(bounding_box_normalized[0] * width)
        # top = int(bounding_box_normalized[1] * height)
        # right = int(bounding_box_normalized[2] * width)
        # bottom = int(bounding_box_normalized[3] * height)
        left = bounding_box[0]
        top = bounding_box[1]
        right = bounding_box[2]
        bottom = bounding_box[3]
        # Bounding Box, normalized
        bounding_box_normalized = np.array([bounding_box[0] / width,
                                            bounding_box[1] / height,
                                            bounding_box[2] / width,
                                            bounding_box[3] / height])
        # Crop
        cropped_image = image[top:bottom, left:right]
        # Hu moments
        moments = cv2.moments(cropped_image, binaryImage=True)
        hu_moments = cv2.HuMoments(moments)[:, 0]
        # Feature
        feature_vector = np.concatenate([bounding_box_normalized, hu_moments])
        return feature_vector

    @staticmethod
    def load_gmm_model(output_path):
        with open(os.path.join(output_path, "gmm_model.sav"), "rb") as f:
            model = pickle.load(f)
        return model

    def localize_text_from_image_path(self, path, unet, max_connected_component_count,
                                      verbose=False):
        # Give the image path to unet
        mask = unet.get_result_images(visuals_path=None, paths=[path], batch_size=1, verbose=False)
        mask = mask[0]
        # We have the segmented image from the unet. Apply connected components analysis first.
        output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        # if numLabels > 2:
        #     print("X")
        # Score all individual components
        bb_list = []
        score_list = []
        features_for_components = []
        for label_id in range(numLabels):
            if label_id == 0:
                continue
            # Left, top, right, bottom
            cc_bounding_box = np.array(
                [stats[label_id, 0],
                 stats[label_id, 1],
                 stats[label_id, 0] + stats[label_id, 2],
                 stats[label_id, 1] + stats[label_id, 3]])
            bb_list.append(cc_bounding_box)
            feature_vector = self.get_gmm_feature_from_bounding_box(image=mask, bounding_box=cc_bounding_box)
            features_for_components.append(feature_vector)
        features_for_components = np.stack(features_for_components, axis=0)
        scores = self.gmm.score_samples(features_for_components)
        scores_sorted_ids = np.argsort(scores)[::-1][:max_connected_component_count]

        # Get all subsets of the scores_sorted_ids
        all_subsets = Utils.powerset(scores_sorted_ids)
        # Score all subsets
        subset_scores = []
        subset_features = []
        subset_bounding_boxes = []
        for subset in all_subsets:
            if len(subset) == 0:
                continue
            # BB of the union
            left = np.inf
            top = np.inf
            right = -np.inf
            bottom = -np.inf
            for cc_id in subset:
                left = min(left, bb_list[cc_id][0])
                top = min(top, bb_list[cc_id][1])
                right = max(right, bb_list[cc_id][2])
                bottom = max(bottom, bb_list[cc_id][3])
            union_bounding_box = np.array([left, top, right, bottom])
            feature_vector = self.get_gmm_feature_from_bounding_box(image=mask, bounding_box=union_bounding_box)
            subset_bounding_boxes.append(union_bounding_box)
            subset_features.append(feature_vector)
        subset_features = np.stack(subset_features, axis=0)
        subset_scores = self.gmm.score_samples(subset_features)
        selected_id = np.argmax(subset_scores)
        selected_bounding_box_for_text = subset_bounding_boxes[selected_id]
        if verbose and numLabels > 2:
            mask_copy = mask.copy()
            mask_bgr = cv2.cvtColor(255 * mask_copy, cv2.COLOR_GRAY2BGR)
            left = int(selected_bounding_box_for_text[0])
            top = int(selected_bounding_box_for_text[1])
            right = int(selected_bounding_box_for_text[2])
            bottom = int(selected_bounding_box_for_text[3])
            cv2.rectangle(mask_bgr, (left, top), (right, bottom), color=(0, 0, 255))
            cv2.imshow("Mask", 255 * mask_copy)
            cv2.imshow("Mask with Detection", mask_bgr)
            cv2.waitKey(0)
        return selected_bounding_box_for_text, mask
