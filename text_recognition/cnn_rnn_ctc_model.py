import tensorflow as tf
import numpy as np
import cv2
import time
import os
import shutil
import pickle
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import classification_report
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

# from blaze_face.align_bb_data import AlignBBData
# from blaze_face.get_predicted_boxes import GetPredictedBoxes
# from blaze_face.multibox_loss import MultiboxLoss
# from blaze_face.nms_layer import NmsLayer
# from blaze_face.ssd_augmentation import SsdAugmentation
from text_detection.align_bb_data import AlignBBData
from text_detection.get_predicted_boxes import GetPredictedBoxes
from text_detection.multibox_loss import MultiboxLoss
from text_detection.ssd_augmentation import SsdAugmentation
from utils import Utils


class CnnRnnCtcModel:
    def __init__(self, model_path, model_name, input_height, l2_lambda,
                 horizontal_delta_ratio=0.15,
                 vertical_delta_ratio=0.05,
                 delta_probability=0.3,
                 brightness_max_delta=0.25,
                 contrast_range=(0.9, 1.1),
                 hue_max_delta=0.25,
                 saturation_range=(0.8, 1.2)
                 ):
        self.inputHeight = input_height
        self.modelPath = model_path
        self.modelName = model_name
        self.l2Lambda = l2_lambda
        self.horizontalDeltaRatio = horizontal_delta_ratio
        self.verticalDeltaRatio = vertical_delta_ratio
        self.deltaProbability = delta_probability
        self.tweakProbability = 1.0 - np.power(1.0 - self.deltaProbability, 0.25)
        self.brightness_max_delta = brightness_max_delta
        self.contrast_range = contrast_range
        self.hue_max_delta = hue_max_delta
        self.saturation_range = saturation_range
        # self.imageInput = tf.keras.Input(shape=self.inputShape, name="imageInput")
        # self.maskInput = tf.keras.Input(shape=(self.inputShape[0], self.inputShape[1]), name="maskInput",
        #                                 dtype=tf.int32)
        # self.weightInput = tf.keras.Input(shape=(self.inputShape[0], self.inputShape[1]), name="weightInput")
        # self.logits = None
        # self.loss = None
        # self.layerWidth = layer_width
        # self.layerDepth = layer_depth
        # self.labelCount = label_count
        # self.model = None
        self.imageDict = {}
        self.dataDict = {}
        # self.maskDict = {}
        # self.weightDict = {}
        # self.originalImagesDict = {}
        self.accuracyMetric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.lossTracker = tf.keras.metrics.Mean()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def tweak_bounding_box(self, image, gt_bounding_box):
        image_width = image.shape.as_list()[1]
        image_height = image.shape.as_list()[0]
        bb_width = gt_bounding_box[2] - gt_bounding_box[0]
        bb_height = gt_bounding_box[3] - gt_bounding_box[1]
        # Step 1): Apply random bb size tweaking
        # Left
        delta_left = int(np.random.uniform() < self.tweakProbability) * int(
            np.random.uniform(low=0.0, high=self.horizontalDeltaRatio * bb_width))
        left = gt_bounding_box[0] - delta_left
        # Top
        delta_top = int(np.random.uniform() < self.tweakProbability) * int(
            np.random.uniform(low=0.0, high=self.verticalDeltaRatio * bb_height))
        top = gt_bounding_box[1] - delta_top
        # Left
        delta_right = int(np.random.uniform() < self.tweakProbability) * int(
            np.random.uniform(low=0.0, high=self.horizontalDeltaRatio * bb_width))
        right = gt_bounding_box[2] + delta_right
        # Bottom
        delta_bottom = int(np.random.uniform() < self.tweakProbability) * int(
            np.random.uniform(low=0.0, high=self.verticalDeltaRatio * bb_height))
        bottom = gt_bounding_box[3] + delta_bottom
        # Step 2): Fit into image size
        # if delta_left != 0 or delta_top != 0 or delta_right != 0 or delta_bottom != 0:
        #     print("X")
        left = max(0, left)
        top = max(0, top)
        right = min(image_width, right)
        bottom = min(image_height, bottom)
        gt_bounding_box = np.array([left, top, right, bottom])
        return gt_bounding_box

    def apply_rotation_normalization(self, image):
        image_rgb = image.numpy()
        image_grayscale = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2GRAY)
        ret2, thresholded_image = cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        black_coords = np.nonzero(thresholded_image == 0)
        black_coords = np.stack([black_coords[0], black_coords[1]], axis=1).astype(np.float32)
        pca = PCA()
        pca.fit(black_coords)
        components = pca.components_.copy()
        components[0] = np.sign(components[0, 1]) * components[0]
        components[1] = np.sign(components[1, 0]) * components[1]
        angle = np.rad2deg(np.arccos(np.clip(components[0, 1], -1.0, 1.0)))
        if components[0, 0] >= 0:
            angle = -angle
        r = R.from_euler('Z', angle, degrees=True).as_matrix()
        rotated_image_rgb = cv2.warpAffine(image_rgb, r[0:2, :], (image_rgb.shape[1], image_rgb.shape[0]))
        return rotated_image_rgb

    def get_input_from_path(self, file_path, augment, verbose):
        image_path = file_path.numpy().decode("utf-8")
        root_path = os.path.split(image_path)[0]
        file_name = os.path.split(image_path)[1]
        if image_path not in self.imageDict:
            assert image_path not in self.imageDict and image_path not in self.dataDict
            image = tf.io.decode_png(tf.io.read_file(file_path))
            detection_stats_path = os.path.join(root_path, file_name[:-4] + "_detection_stats.dat")
            with open(detection_stats_path, "rb") as f:
                detections_dict = pickle.load(f)
            self.imageDict[image_path] = image
            self.dataDict[image_path] = detections_dict
        else:
            image = self.imageDict[image_path]
            detections_dict = self.dataDict[image_path]
        gt_bounding_box = np.array(
            [detections_dict["bounding_box"][0, 1],
             detections_dict["bounding_box"][0, 0],
             detections_dict["bounding_box"][1, 1],
             detections_dict["bounding_box"][1, 0]])
        if augment:
            # Step 1) Tweak bounding box boundaries randomly as a part of the augmentation
            gt_bounding_box = self.tweak_bounding_box(image=image, gt_bounding_box=gt_bounding_box)
        cropped_image = image[gt_bounding_box[1]:gt_bounding_box[3], gt_bounding_box[0]:gt_bounding_box[2]]
        rotation_normalized_image = self.apply_rotation_normalization(image=cropped_image)
        if augment:
            # Step 2) Apply brightness - contrast augmentations
            augmented_image = tf.image.random_brightness(rotation_normalized_image, self.brightness_max_delta)
            augmented_image = tf.image.random_contrast(augmented_image,
                                                       lower=self.contrast_range[0],
                                                       upper=self.contrast_range[1])
        else:
            augmented_image = rotation_normalized_image
        new_width = int((self.inputHeight / augmented_image.shape.as_list()[0]) * augmented_image.shape.as_list()[1])
        final_image = tf.cast(tf.image.resize(augmented_image, size=(self.inputHeight, new_width)), dtype=tf.uint8)
        if verbose:
            cv2.imshow("cropped_image", cv2.cvtColor(cropped_image.numpy(), cv2.COLOR_RGB2BGR))
            cv2.imshow("rotation_normalized_image", cv2.cvtColor(rotation_normalized_image, cv2.COLOR_RGB2BGR))
            cv2.imshow("augmented_image", cv2.cvtColor(augmented_image.numpy(), cv2.COLOR_RGB2BGR))
            cv2.imshow("final_image", cv2.cvtColor(final_image.numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
        return final_image, file_name[:-4]

    def get_batch_data(self, batch_paths, verbose):
        images = []
        barcodes = []
        for img_id, img_path in enumerate(batch_paths):
            image, barcode = self.get_input_from_path(file_path=img_path, augment=True, verbose=False)
            images.append(image)
            barcodes.append(barcode)
        heights = [img.shape[0] for img in images]
        widths = [img.shape[1] for img in images]
        assert len(set(heights)) == 1 and heights[0] == self.inputHeight
        max_width = np.max(np.array(widths))
        images_padded = []
        for img in images:
            img_padded = tf.image.pad_to_bounding_box(img, 0, 0, target_height=self.inputHeight, target_width=max_width)
            if verbose:
                cv2.imshow("img_padded", cv2.cvtColor(img_padded.numpy(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
            images_padded.append(img_padded)
        images_tensor = tf.stack(images_padded, axis=0)
        print("X")

    def train(self, train_paths, test_paths, batch_size, epoch_count):
        root_path = os.path.join(self.modelPath, self.modelName)
        Utils.create_directory(path=root_path)

        train_iterator = \
            tf.data.Dataset.from_tensor_slices(train_paths).shuffle(buffer_size=100).batch(
                batch_size=batch_size)

        with tf.device("GPU"):
            for epoch_id in range(epoch_count):
                for iter_id, batch_paths in enumerate(train_iterator):
                    t0 = time.time()
                    images = []
                    barcodes = []
                    self.get_batch_data(batch_paths=batch_paths, verbose=True)
                    # for img_id, img_path in enumerate(batch_paths):
                    #     self.get_batch_data()
                        # image, barcode = self.get_input_from_path(file_path=img_path, augment=True, verbose=False)
                        # images.append(image)
                        # barcodes.append(barcode)
                    t1 = time.time()
