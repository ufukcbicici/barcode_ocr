import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import cv2
import pickle
from scipy.spatial import distance_matrix


class BoundingBoxGMMModeller(object):
    def __init__(self, cluster_count):
        self.clusterCount = cluster_count

    def train_gmm_model(self, visual_paths, train_paths, unet, batch_size, dilation_kernel_size, verbose=True):
        # iterator = \
        #     tf.data.Dataset.from_tensor_slices(train_paths).shuffle(buffer_size=100).batch(batch_size=batch_size)
        # mask_list = unet.get_result_images(visuals_path=visual_paths, paths=train_paths,
        #                                    batch_size=batch_size, verbose=True)
        #
        # print("X")
        for path in tqdm(train_paths):
            root_path = os.path.split(path)[0]
            file_name = os.path.split(path)[1]
            mask_image_path = os.path.join(root_path, file_name[:-4] + "_mask.png")
            detection_stats_path = os.path.join(root_path, file_name[:-4] + "_detection_stats.dat")
            with open(detection_stats_path, "rb") as f:
                detections_dict = pickle.load(f)
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
            output = cv2.connectedComponentsWithStats(dilated_mask_image, 8, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = output
            # The connected component bounding boxes; compare with the ground truth box; fix the most likely one
            # bu choosing the most similar bounding box
            bb_list = []
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
            bb_list = np.stack(bb_list, axis=0)
            gt_bounding_box = np.array(
                [detections_dict["bounding_box"][0, 1],
                 detections_dict["bounding_box"][0, 0],
                 detections_dict["bounding_box"][1, 1],
                 detections_dict["bounding_box"][1, 0]])
            distances = distance_matrix(x=bb_list, y=np.expand_dims(gt_bounding_box, axis=0))
            print("X")



