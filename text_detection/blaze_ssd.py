import tensorflow as tf
import numpy as np
import cv2
import time
import os
import shutil
import pickle

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


class BlazeSsdDetector:
    def __init__(self,
                 model_name,
                 model_path,
                 input_shape,
                 anchor_boxes=(2, 6),
                 brightness_max_delta=0.25,
                 contrast_range=(0.9, 1.1),
                 hue_max_delta=0.25,
                 saturation_range=(0.8, 1.2)
                 ):
        self.modelName = model_name
        self.modelPath = model_path
        self.inputShape = input_shape
        self.anchorBoxes = anchor_boxes
        self.brightness_max_delta = brightness_max_delta
        self.contrast_range = contrast_range
        self.hue_max_delta = hue_max_delta
        self.saturation_range = saturation_range
        self.backboneModel = self.network(input_shape=self.inputShape)
        self.confidenceOutput = None
        self.bbRegressionOutput = None
        self.alignBBDataLayer = None
        self.detectorModel = None
        # self.priorBoxes = prior_boxes
        # self.priorBoxes = sorted(self.priorBoxes, key=lambda bb: bb[0] * bb[1])
        # self.priorBoxes = np.stack(self.priorBoxes, axis=0)
        # assert sum(self.anchorBoxes) == self.priorBoxes.shape[0]
        self.priorBoxTensors = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.lossTracker = tf.keras.metrics.Mean(name="loss_tracker")

    def channel_padding(self, x):
        """
        zero padding in an axis of channel
        """

        return tf.keras.backend.concatenate([x, tf.zeros_like(x)], axis=-1)

    def single_blaze_block(self, x, filters=24, kernel_size=5, strides=1, padding='same'):
        # depth-wise separable convolution
        x_0 = tf.keras.layers.SeparableConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False)(x)

        x_1 = tf.keras.layers.BatchNormalization()(x_0)

        # Residual connection

        if strides == 2:
            input_channels = x.shape[-1]
            output_channels = x_1.shape[-1]

            x_ = tf.keras.layers.MaxPooling2D()(x)

            if output_channels - input_channels != 0:
                # channel padding
                x_ = tf.keras.layers.Lambda(self.channel_padding)(x_)

            out = tf.keras.layers.Add()([x_1, x_])
            return tf.keras.layers.Activation("relu")(out)

        out = tf.keras.layers.Add()([x_1, x])
        return tf.keras.layers.Activation("relu")(out)

    def double_blaze_block(self, x, filters_1=24, filters_2=96,
                           kernel_size=5, strides=1, padding='same'):
        # depth-wise separable convolution, project
        x_0 = tf.keras.layers.SeparableConv2D(
            filters=filters_1,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False)(x)

        x_1 = tf.keras.layers.BatchNormalization()(x_0)

        x_2 = tf.keras.layers.Activation("relu")(x_1)

        # depth-wise separable convolution, expand
        x_3 = tf.keras.layers.SeparableConv2D(
            filters=filters_2,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            use_bias=False)(x_2)

        x_4 = tf.keras.layers.BatchNormalization()(x_3)

        # Residual connection

        if strides == 2:
            input_channels = x.shape[-1]
            output_channels = x_4.shape[-1]

            x_ = tf.keras.layers.MaxPooling2D()(x)

            if output_channels - input_channels != 0:
                # channel padding
                x_ = tf.keras.layers.Lambda(self.channel_padding)(x_)

            out = tf.keras.layers.Add()([x_4, x_])
            return tf.keras.layers.Activation("relu")(out)

        out = tf.keras.layers.Add()([x_4, x])
        return tf.keras.layers.Activation("relu")(out)

    def network(self, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)

        x_0 = tf.keras.layers.Conv2D(
            filters=24, kernel_size=5, strides=2, padding='same')(inputs)
        x_0 = tf.keras.layers.BatchNormalization()(x_0)
        x_0 = tf.keras.layers.Activation("relu")(x_0)

        # single BlazeBlock phase
        x_1 = self.single_blaze_block(x_0)
        x_2 = self.single_blaze_block(x_1)
        x_3 = self.single_blaze_block(x_2, strides=2, filters=48)
        x_4 = self.single_blaze_block(x_3, filters=48)
        x_5 = self.single_blaze_block(x_4, filters=48)

        # double BlazeBlock phase

        x_6 = self.double_blaze_block(x_5, strides=2)
        x_7 = self.double_blaze_block(x_6)
        x_8 = self.double_blaze_block(x_7)
        x_9 = self.double_blaze_block(x_8, strides=2)
        x10 = self.double_blaze_block(x_9)
        x11 = self.double_blaze_block(x10)

        model = tf.keras.models.Model(inputs=inputs, outputs=[x_8, x11])
        return model

    def convert_from_image_coords_to_normalized_coords(self, image_shape, image_coords,
                                                       image=None, verbose=False):
        curr_ratio = image_shape[0] / image_shape[1]
        if curr_ratio <= self.inputShape[0] / self.inputShape[1]:
            resize_ratio = self.inputShape[1] / image_shape[1]
            resized_shape = [int(resize_ratio * image_shape[0]), self.inputShape[1]]
        else:
            resize_ratio = self.inputShape[0] / image_shape[0]
            resized_shape = [self.inputShape[0], int(resize_ratio * image_shape[1])]
        image_coords_resized = resize_ratio * image_coords
        left = image_coords_resized[0]
        top = image_coords_resized[1]
        right = image_coords_resized[2]
        bottom = image_coords_resized[3]
        image_coords_normalized = np.array(
            [left / self.inputShape[1], top / self.inputShape[0],
             right / self.inputShape[1], bottom / self.inputShape[0]])
        if verbose:
            assert image is not None
            image_resized = tf.image.resize(image, size=resized_shape)
            image_resized_padded = tf.image.pad_to_bounding_box(image_resized, 0, 0, self.inputShape[0],
                                                                self.inputShape[1])
            image_resized_padded_bgr = tf.cast(image_resized_padded, tf.uint8).numpy()
            left_verbose = int(image_coords_normalized[0] * self.inputShape[1])
            top_verbose = int(image_coords_normalized[1] * self.inputShape[0])
            right_verbose = int(image_coords_normalized[2] * self.inputShape[1])
            bottom_verbose = int(image_coords_normalized[3] * self.inputShape[0])
            cv2.rectangle(image_resized_padded_bgr, (left_verbose, top_verbose),
                          (right_verbose, bottom_verbose), color=(0, 255, 0))
            cv2.imshow("Normalized Coords", image_resized_padded_bgr)
            cv2.waitKey(0)
        return image_coords_normalized

    def cluster_bounding_boxes(self, bounding_box_dict):
        cluster_count = sum(self.anchorBoxes)
        bb_coords = []
        for file_path in tqdm(bounding_box_dict.keys()):
            image = cv2.imread(file_path)
            bb_image_coords_list = bounding_box_dict[file_path]
            for bb_image_coords in bb_image_coords_list:
                bb_normalized_coords = \
                    self.convert_from_image_coords_to_normalized_coords(image_shape=image.shape,
                                                                        image_coords=bb_image_coords,
                                                                        image=image,
                                                                        verbose=False)
                bb_coords.append(bb_normalized_coords)
        bb_coords = np.stack(bb_coords, axis=0)
        kmedoids = KMedoids(n_clusters=cluster_count)
        widths = bb_coords[:, 2] - bb_coords[:, 0]
        heights = bb_coords[:, 3] - bb_coords[:, 1]
        bb_dimensions = np.stack([widths, heights], axis=1)
        kmedoids.fit(X=bb_dimensions)
        counter = Counter(kmedoids.labels_)
        print("counter={0}".format(counter))
        return kmedoids.cluster_centers_

    def build_detector(self, bounding_box_dict):
        # Confidence outputs for anchor boxes
        self.confidenceOutput = []
        assert len(self.backboneModel.output) == len(self.anchorBoxes)
        for i in range(len(self.backboneModel.output)):
            bb_conf = tf.keras.layers.Conv2D(filters=self.anchorBoxes[i],
                                             kernel_size=3,
                                             padding='same')(self.backboneModel.output[i])
            # output_shape = bb_conf.shape
            # flat_dim = np.prod(output_shape[1:])
            # bb_conf_reshaped = tf.keras.layers.Reshape((flat_dim, 1))(bb_conf)
            self.confidenceOutput.append(bb_conf)
        # self.confidenceOutput = tf.keras.layers.Concatenate(axis=1)(confidence_outputs)

        # Bounding box regression outputs for anchor boxes - [x, y, w, h]
        self.bbRegressionOutput = []
        for i in range(len(self.backboneModel.output)):
            bb_loc = tf.keras.layers.Conv2D(filters=self.anchorBoxes[i] * 4,
                                            kernel_size=3,
                                            padding='same')(self.backboneModel.output[i])
            # output_shape = bb_loc.shape
            # flat_dim = int(np.prod(output_shape[1:]) / 4)
            # bb_loc_reshaped = tf.keras.layers.Reshape((flat_dim, 4))(bb_loc)
            self.bbRegressionOutput.append(bb_loc)
        # self.bbRegressionOutput = tf.keras.layers.Concatenate(axis=1)(bb_regression_outputs)
        # output_combined = tf.keras.layers.Concatenate(axis=-1)([self.confidenceOutput, self.bbRegressionOutput])

        # Determine the prior boxes, for each detector output location in the corresponding feature maps
        prior_boxes = self.cluster_bounding_boxes(bounding_box_dict=bounding_box_dict)
        self.priorBoxTensors = []
        output_indices = []
        for output_id, anchor_count in enumerate(self.anchorBoxes):
            output_indices.extend([output_id] * anchor_count)

        assert len(output_indices) == prior_boxes.shape[0]
        for anchor_id, prior in enumerate(prior_boxes):
            # Select the respective backbone output for this anchor
            curr_output_id = output_indices[anchor_id]
            output_shape = self.backboneModel.output[curr_output_id].shape
            # Calculate each prior corresponding to a feature map location
            w = output_shape[2]
            h = output_shape[1]
            priors_tensor = np.zeros(shape=(h, w, 4))
            for pi in range(h):
                for pj in range(w):
                    center_i = (0.5 + pi) / h
                    center_j = (0.5 + pj) / w
                    left = center_j - 0.5 * prior[0]
                    top = center_i - 0.5 * prior[1]
                    right = center_j + 0.5 * prior[0]
                    bottom = center_i + 0.5 * prior[1]
                    priors_tensor[pi, pj] = np.array([left, top, right, bottom])
            self.priorBoxTensors.append(tf.constant(priors_tensor))

        # Prepare the output of the detector model
        # X = tf.stack(batch_images, axis=0)
        self.alignBBDataLayer = AlignBBData(prior_box_tensors=self.priorBoxTensors)
        confidences_reshaped, regression_reshaped, prior_boxes_reshaped = \
            self.alignBBDataLayer((self.confidenceOutput, self.bbRegressionOutput))
        self.detectorModel = tf.keras.models.Model(
            inputs=self.backboneModel.input,
            outputs=[confidences_reshaped, regression_reshaped, prior_boxes_reshaped])

    def save_model(self, path):
        self.detectorModel.save(path)

    def load_model(self, path):
        self.detectorModel = tf.keras.models.load_model(path, custom_objects={'channel_padding': self.channel_padding})
        # self.detectorModel.load_weights(path)
        print("X")

    def decode_and_augment_images(self, file_path, bb_list, augment, verbose=False):
        raw_file = tf.io.read_file(file_path)
        image = tf.io.decode_png(raw_file)
        original_shape = image.shape.as_list()
        if augment:
            image = tf.image.random_brightness(image, self.brightness_max_delta)
            image = tf.image.random_contrast(image, lower=self.contrast_range[0], upper=self.contrast_range[1])
        # image = tf.image.random_hue(image, max_delta=self.hue_max_delta)
        # image = tf.image.random_saturation(image, lower=self.saturation_range[0], upper=self.saturation_range[1])
        # image = tf.cast(image, tf.float32)
        # image = tf.image.stateless_random_brightness(image, self.brightness_max_delta)
        curr_ratio = image.shape.as_list()[0] / image.shape.as_list()[1]
        if curr_ratio <= self.inputShape[0] / self.inputShape[1]:
            resize_ratio = self.inputShape[1] / image.shape.as_list()[1]
            new_size = [int(resize_ratio * image.shape.as_list()[0]), self.inputShape[1]]
            image_resized = tf.image.resize(image, size=new_size)
        else:
            resize_ratio = self.inputShape[0] / image.shape.as_list()[0]
            new_size = [self.inputShape[0], int(resize_ratio * image.shape.as_list()[1])]
            image_resized = tf.image.resize(image, size=new_size)
        # image_resized_padded = \
        #     tf.image.resize_with_pad(image, target_height=self.inputShape[0], target_width=self.inputShape[1])
        # image_resized_padded = tf.zeros(shape=self.inputShape, dtype=tf.uint8)
        # image_resized_padded[0:image_resized.shape[0], 0:image_resized.shape[1], :] = image_resized
        image_resized_padded = tf.image.pad_to_bounding_box(image_resized, 0, 0, self.inputShape[0],
                                                            self.inputShape[1])
        # Adjust bb coordinates
        adjusted_bb_list = []
        for bb in bb_list:
            normalized_bb = self.convert_from_image_coords_to_normalized_coords(
                image_shape=original_shape, image_coords=bb)
            adjusted_bb_list.append(normalized_bb)
        # Comment out for visualization
        if verbose:
            image_rgb = image.numpy()
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Augmented Image", image_bgr)

            image_resized_padded_rgb = tf.cast(image_resized_padded, tf.uint8).numpy()
            image_resized_padded_bgr = cv2.cvtColor(image_resized_padded_rgb, cv2.COLOR_RGB2BGR)
            for bb in adjusted_bb_list:
                h = image_resized_padded.shape.as_list()[0]
                w = image_resized_padded.shape.as_list()[1]
                left = int(bb[0] * w)
                top = int(bb[1] * h)
                right = int(bb[2] * w)
                bottom = int(bb[3] * h)
                cv2.rectangle(image_resized_padded_bgr, (left, top), (right, bottom), color=(0, 255, 0))
            cv2.imshow("Augmented Image with Resize and Pad", image_resized_padded_bgr)
            cv2.waitKey(0)

        image_resized_padded = (1.0 / 255.0) * image_resized_padded
        return image_resized_padded, adjusted_bb_list, resize_ratio

    def train(self, training_set, batch_size, epoch_count, test_ratio=0.1):
        paths = []
        for path in training_set.keys():
            paths.append(path)
        root_path = os.path.join(self.modelPath, self.modelName)
        Utils.create_directory(path=root_path)

        # for i in range(100):
        #     self.decode_and_augment_images(tpl=dataset[0])

        # Apply train test split
        train_paths, test_paths, = train_test_split(paths, test_size=test_ratio)
        training_images_path = os.path.join(root_path, "training_images.sav")
        test_images_path = os.path.join(root_path, "test_images.sav")
        with open(training_images_path, "wb") as f:
            pickle.dump(train_paths, f)
        with open(test_images_path, "wb") as f:
            pickle.dump(test_paths, f)
        # Save training - test set for later evaluation.
        train_iterator = \
            tf.data.Dataset.from_tensor_slices(train_paths).shuffle(buffer_size=100).batch(batch_size=batch_size)
        # test_set = tf.data.Dataset.from_tensor_slices(test_images). \
        #     shuffle(buffer_size=100).batch(batch_size=batch_size)

        with tf.device("GPU"):
            m_box_loss = MultiboxLoss()
            for epoch_id in range(epoch_count):
                self.lossTracker.reset_states()
                for iter_id, batch_paths in enumerate(train_iterator):
                    batch_images = []
                    batch_bb_list = []
                    # Prepare augmented, resized and padded images and the corresponding bounding boxes
                    t0 = time.time()
                    for file_path_tf in batch_paths:
                        file_path = file_path_tf.numpy().decode("utf-8")
                        bb_list = training_set[file_path]
                        image, image_bb_list, _ = self.decode_and_augment_images(file_path=file_path,
                                                                                 bb_list=bb_list,
                                                                                 augment=True,
                                                                                 verbose=False)
                        batch_images.append(image)
                        batch_bb_list.append(image_bb_list)
                    t1 = time.time()
                    with tf.GradientTape() as tape:
                        X = tf.stack(batch_images, axis=0)
                        confidences_reshaped, regression_reshaped, prior_boxes_reshaped = \
                            self.detectorModel(inputs=X, training=True)
                        total_loss = m_box_loss((confidences_reshaped, regression_reshaped,
                                                 prior_boxes_reshaped, batch_bb_list))
                    grads = tape.gradient(total_loss, self.detectorModel.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.detectorModel.trainable_variables))
                    self.lossTracker.update_state(total_loss)
                    t2 = time.time()
                    print("Total Loss:{0}".format(self.lossTracker.result().numpy()))
                    print("Id:{0} Time:{1}-{2}".format(iter_id, t1 - t0, t2 - t1))
                self.save_model(path=os.path.join(root_path, "model_epoch{0}".format(epoch_id)))

    def eval_images(self,
                    training_set,
                    image_paths,
                    batch_size,
                    visual_results_path,
                    txt_results_path,
                    positive_threshold=0.5,
                    iou_threshold=0.5):
        with tf.device("GPU"):
            test_iterator = tf.data.Dataset.from_tensor_slices(image_paths).batch(batch_size=batch_size)
            get_predicted_boxes = GetPredictedBoxes()
            Utils.create_directory(path=visual_results_path)
            Utils.create_directory(path=txt_results_path)
            detection_ground_truths_path = os.path.join(txt_results_path, "ground_truth")
            detection_predictions_path = os.path.join(txt_results_path, "predictions")
            Utils.create_directory(path=detection_ground_truths_path)
            Utils.create_directory(path=detection_predictions_path)
            detection_times = []
            for iter_id, batch_paths in enumerate(test_iterator):
                batch_images = []
                batch_bb_list = []
                resize_ratios = []
                # Prepare augmented, resized and padded images and the corresponding bounding boxes
                t0 = time.time()
                for file_path_tf in batch_paths:
                    file_path = file_path_tf.numpy().decode("utf-8")
                    bb_list = training_set[file_path]
                    image, image_bb_list, resize_ratio = self.decode_and_augment_images(file_path=file_path,
                                                                                        bb_list=bb_list,
                                                                                        augment=False,
                                                                                        verbose=False)
                    batch_images.append(image)
                    batch_bb_list.append(image_bb_list)
                    resize_ratios.append(resize_ratio)
                t1 = time.time()
                X = tf.stack(batch_images, axis=0)
                confidences_reshaped, regression_reshaped, prior_boxes_reshaped = \
                    self.detectorModel(inputs=X, training=False)
                # Build predicted box locations
                predicted_boxes = get_predicted_boxes((regression_reshaped, prior_boxes_reshaped))
                t1 = time.time()
                print("t1 - t0:{0}".format(t1 - t0))
                detection_times.append(t1 - t0)
                for image_id in range(batch_paths.shape[0]):
                    img_path = batch_paths[image_id].numpy().decode("utf-8")
                    image = cv2.imread(img_path)
                    conf = confidences_reshaped[image_id]
                    conf = conf[:, 0]
                    conf = tf.sigmoid(conf)
                    box_locations = predicted_boxes[image_id]
                    box_locations_yx = \
                        tf.stack([box_locations[:, 1], box_locations[:, 0], box_locations[:, 3], box_locations[:, 2]],
                                 axis=-1)
                    selected_indices = tf.image.non_max_suppression(box_locations_yx, conf, max_output_size=10,
                                                                    iou_threshold=iou_threshold,
                                                                    score_threshold=positive_threshold)
                    box_predictions = tf.gather(box_locations, selected_indices)
                    box_locations_np = box_predictions.numpy()
                    selected_scores = tf.gather(conf, selected_indices)
                    selected_scores_np = selected_scores.numpy()
                    img_result = image.copy()
                    resize_ratio = resize_ratios[image_id]
                    gt_bb_list = training_set[img_path]
                    for box in box_locations_np:
                        left = int((box[0] * self.inputShape[1]) / resize_ratio)
                        top = int((box[1] * self.inputShape[0]) / resize_ratio)
                        right = int((box[2] * self.inputShape[1]) / resize_ratio)
                        bottom = int((box[3] * self.inputShape[0]) / resize_ratio)
                        cv2.rectangle(img_result, (left, top), (right, bottom), color=(0, 255, 0))
                    image_file_name = os.path.split(batch_paths[image_id].numpy().decode("utf-8"))[1]
                    image_file_path = os.path.join(visual_results_path, image_file_name)
                    cv2.imwrite(filename=image_file_path, img=img_result)
                    # Text outputs
                    ground_truth_file_path = os.path.join(detection_ground_truths_path, image_file_name[:-4] + ".txt")
                    prediction_file_path = os.path.join(detection_predictions_path, image_file_name[:-4] + ".txt")
                    with open(ground_truth_file_path, 'a') as f:
                        for bb in gt_bb_list:
                            f.write("{0} {1} {2} {3} {4}\n".format(
                                "barcode",
                                int(bb[0]),
                                int(bb[1]),
                                int(bb[2]),
                                int(bb[3])))
                    # Predictions
                    with open(prediction_file_path, 'a') as f:
                        for idx in range(box_locations_np.shape[0]):
                            f.write("{0} {1} {2} {3} {4} {5}\n".format(
                                "barcode",
                                selected_scores_np[idx],
                                int((box[0] * self.inputShape[1]) / resize_ratio),
                                int((box[1] * self.inputShape[0]) / resize_ratio),
                                int((box[2] * self.inputShape[1]) / resize_ratio),
                                int((box[3] * self.inputShape[0]) / resize_ratio)))


if __name__ == "__main__":
    pass
