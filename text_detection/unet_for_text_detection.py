import tensorflow as tf
import numpy as np
import cv2
import time
import os
import shutil
import pickle

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


class Unet:
    def __init__(self, model_path, model_name, input_shape, layer_width, layer_depth, label_count,
                 dilation_kernel,
                 class_weights,
                 l2_lambda,
                 brightness_max_delta=0.25,
                 contrast_range=(0.9, 1.1),
                 hue_max_delta=0.25,
                 saturation_range=(0.8, 1.2)
                 ):
        self.inputShape = input_shape
        self.modelPath = model_path
        self.modelName = model_name
        self.dilationKernel = dilation_kernel
        self.l2Lambda = l2_lambda
        self.classWeights = class_weights
        self.brightness_max_delta = brightness_max_delta
        self.contrast_range = contrast_range
        self.hue_max_delta = hue_max_delta
        self.saturation_range = saturation_range
        self.imageInput = tf.keras.Input(shape=self.inputShape, name="imageInput")
        self.maskInput = tf.keras.Input(shape=(self.inputShape[0], self.inputShape[1]), name="maskInput",
                                        dtype=tf.int32)
        self.weightInput = tf.keras.Input(shape=(self.inputShape[0], self.inputShape[1]), name="weightInput")
        self.logits = None
        self.loss = None
        self.layerWidth = layer_width
        self.layerDepth = layer_depth
        self.labelCount = label_count
        self.model = None
        self.imageDict = {}
        self.maskDict = {}
        self.weightDict = {}
        self.originalImagesDict = {}
        self.accuracyMetric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.lossTracker = tf.keras.metrics.Mean()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def build_network(self):
        tf_img, tf_msk, tf_weights = self.imageInput, self.maskInput, self.weightInput
        # Entry
        net = tf_img / 127.5 - 1
        # Contracting Part
        conv1, pool1 = self.conv_conv_pool(net, [8 * self.layerWidth] * self.layerDepth, name=1)
        conv2, pool2 = self.conv_conv_pool(pool1, [16 * self.layerWidth] * self.layerDepth, name=2)
        conv3, pool3 = self.conv_conv_pool(pool2, [32 * self.layerWidth] * self.layerDepth, name=3)
        conv4, pool4 = self.conv_conv_pool(pool3, [64 * self.layerWidth] * self.layerDepth, name=4)
        conv5, _ = self.conv_conv_pool(pool4, [128 * self.layerWidth] * self.layerDepth, name=5, pool=False)
        # up_conv = self.upconv_2D(conv5, 64, name=10)
        # Expanding Part
        up6 = self.upconv_concat(conv5, conv4, 64 * self.layerWidth, name=6)
        conv6, _ = self.conv_conv_pool(up6, [64 * self.layerWidth] * self.layerDepth, name=6, pool=False)
        up7 = self.upconv_concat(conv6, conv3, 32 * self.layerWidth, name=7)
        conv7, _ = self.conv_conv_pool(up7, [32 * self.layerWidth] * self.layerDepth, name=7, pool=False)
        up8 = self.upconv_concat(conv7, conv2, 16 * self.layerWidth, name=8)
        conv8, _ = self.conv_conv_pool(up8, [16 * self.layerWidth] * self.layerDepth, name=8, pool=False)
        up9 = self.upconv_concat(conv8, conv1, 8 * self.layerWidth, name=9)
        conv9, _ = self.conv_conv_pool(up9, [8 * self.layerWidth] * self.layerDepth, name=9, pool=False)
        # Logits
        self.logits = tf.keras.layers.Conv2D(self.labelCount, 1, name='final',
                                             activation=None,
                                             padding='same')(conv9)
        self.model = tf.keras.Model(inputs=self.imageInput, outputs=self.logits)

    # Get conv layer
    def conv_conv_pool(self,
                       input_,
                       n_filters,
                       name,
                       pool=True,
                       activation=tf.nn.relu):
        net = input_
        for i, filter_count in enumerate(n_filters):
            net = tf.keras.layers.Conv2D(filter_count, 3, padding="same", name="conv_{0}".format(name))(net)
            net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.ReLU()(net)
        if pool is False:
            return net, None
        else:
            pooled_net = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(net)
            return net, pooled_net

    def upconv_2D(self, tensor, n_filter, name):
        net = tf.keras.layers.Conv2DTranspose(filters=n_filter, kernel_size=2, strides=2,
                                              name="upsample_{}".format(name))(tensor)
        return net

    def upconv_concat(self, inputA, input_B, n_filter, name):
        up_conv = self.upconv_2D(inputA, n_filter, name)

        return tf.concat(
            [up_conv, input_B], axis=-1, name="concat_{}".format(name))

    def decode_and_augment_images(self, file_path, augment, verbose=False):
        image_path = file_path.numpy().decode("utf-8")
        mask_image_path = image_path[:-4] + "_mask.png"
        if image_path not in self.imageDict:
            assert image_path not in self.maskDict and image_path not in self.weightDict
            image = tf.io.decode_png(tf.io.read_file(file_path))
            self.originalImagesDict[image_path] = image.numpy()
            mask_image = tf.io.decode_png(tf.io.read_file(mask_image_path))
            dilated_mask_image = tf.nn.dilation2d(
                input=tf.expand_dims(mask_image, axis=0),
                filters=tf.zeros(shape=[self.dilationKernel, self.dilationKernel, 1], dtype=tf.uint8),
                strides=[1, 1, 1, 1],
                padding="SAME",
                data_format="NHWC",
                dilations=[1, 1, 1, 1]
            )
            dilated_mask_image = dilated_mask_image[0]
            curr_ratio = image.shape.as_list()[0] / image.shape.as_list()[1]
            if curr_ratio <= self.inputShape[0] / self.inputShape[1]:
                resize_ratio = self.inputShape[1] / image.shape.as_list()[1]
                new_size = [int(resize_ratio * image.shape.as_list()[0]), self.inputShape[1]]
                image_resized = tf.image.resize(image, size=new_size)
                mask_resized = tf.image.resize(mask_image, size=new_size)
                dilated_mask_image_resized = tf.image.resize(dilated_mask_image, size=new_size)
            else:
                resize_ratio = self.inputShape[0] / image.shape.as_list()[0]
                new_size = [self.inputShape[0], int(resize_ratio * image.shape.as_list()[1])]
                image_resized = tf.image.resize(image, size=new_size)
                mask_resized = tf.image.resize(mask_image, size=new_size)
                dilated_mask_image_resized = tf.image.resize(dilated_mask_image, size=new_size)

            dilated_mask_image_resized = tf.cast(tf.greater(dilated_mask_image_resized, 0),
                                                 dtype=tf.int32)
            weight_mask = tf.zeros_like(input=dilated_mask_image_resized, dtype=image_resized.dtype)
            for class_id, class_weight in self.classWeights.items():
                class_weights_arr = tf.where(tf.equal(dilated_mask_image_resized, class_id),
                                             class_weight * tf.ones_like(weight_mask), tf.zeros_like(weight_mask))
                weight_mask += class_weights_arr

            offset_x = self.inputShape[1] - image_resized.shape[1]
            offset_y = self.inputShape[0] - image_resized.shape[0]
            image_resized_padded = tf.image.pad_to_bounding_box(image_resized, offset_y, offset_x, self.inputShape[0],
                                                                self.inputShape[1])
            mask_resized_padded = tf.image.pad_to_bounding_box(mask_resized, offset_y, offset_x, self.inputShape[0],
                                                               self.inputShape[1])
            dilated_mask_image_resized_padded = tf.image.pad_to_bounding_box(dilated_mask_image_resized, offset_y,
                                                                             offset_x,
                                                                             self.inputShape[0],
                                                                             self.inputShape[1])
            weight_mask_padded = tf.image.pad_to_bounding_box(weight_mask, offset_y, offset_x,
                                                              self.inputShape[0],
                                                              self.inputShape[1])
            self.imageDict[image_path] = image_resized_padded
            self.maskDict[image_path] = dilated_mask_image_resized_padded
            self.weightDict[image_path] = weight_mask_padded
        else:
            assert image_path in self.maskDict and image_path in self.weightDict
            image_resized_padded = self.imageDict[image_path]
            dilated_mask_image_resized_padded = self.maskDict[image_path]
            weight_mask_padded = self.weightDict[image_path]

        if augment:
            image_resized_padded = tf.cast(image_resized_padded, dtype=tf.uint8)
            image_resized_padded = tf.image.random_brightness(image_resized_padded, self.brightness_max_delta)
            image_resized_padded = tf.image.random_contrast(image_resized_padded, lower=self.contrast_range[0],
                                                            upper=self.contrast_range[1])
            image_resized_padded = tf.cast(image_resized_padded, dtype=tf.float32)

        if verbose:
            # np_image_resized_rgb = image_resized.numpy().astype(np.uint8)
            # np_image_resized_bgr = cv2.cvtColor(np_image_resized_rgb, cv2.COLOR_RGB2BGR)
            # np_mask_resized = mask_resized.numpy().astype(np.uint8)
            np_image_resized_padded_rgb = image_resized_padded.numpy().astype(np.uint8)
            np_image_resized_padded_bgr = cv2.cvtColor(np_image_resized_padded_rgb, cv2.COLOR_RGB2BGR)
            # np_mask_resized_padded = mask_resized_padded.numpy().astype(np.uint8)
            np_dilated_mask_image_resized_padded = 255 * dilated_mask_image_resized_padded.numpy().astype(np.uint8)
            np_weight_mask_padded = weight_mask_padded.numpy()
            np_weight_mask_colored = np.zeros_like(np_image_resized_padded_bgr)
            for class_id, class_weight in self.classWeights.items():
                color_mask = np.where(np.isclose(np_weight_mask_padded, class_weight),
                                      np.random.randint(low=0, high=256, size=(3,)),
                                      np.array([0, 0, 0])).astype(dtype=np.uint8)
                np_weight_mask_colored += color_mask

            # cv2.imshow("np_image_resized_bgr", np_image_resized_bgr)
            # cv2.imshow("np_mask_resized", np_mask_resized)
            cv2.imshow("np_image_resized_padded_bgr", np_image_resized_padded_bgr)
            # cv2.imshow("np_mask_resized_padded", np_mask_resized_padded)
            cv2.imshow("np_dilated_mask_image_resized_padded", np_dilated_mask_image_resized_padded)
            cv2.imshow("np_weight_mask_colored", np_weight_mask_colored)
            cv2.waitKey(0)
        return image_resized_padded, dilated_mask_image_resized_padded, weight_mask_padded

    def calculate_loss(self, images, masks, weights):
        logits = self.model(images)
        # Pixel-wise cross entropy
        masks = masks[:, :, :, 0]
        weights = weights[:, :, :, 0]
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masks, logits=logits)
        cross_entropy_with_weights = weights * cross_entropy
        ce_loss = tf.reduce_mean(cross_entropy_with_weights)
        # L2 - weight decay
        l2_losses = []
        # for variable in self.model.trainable_variables:
        for variable in self.model.trainable_variables:
            if "kernel" in variable.name:
                l2_losses.append(self.l2Lambda * tf.nn.l2_loss(variable))
        l2_loss = tf.add_n(l2_losses)
        total_loss = ce_loss + l2_loss
        return total_loss

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def eval(self, paths, batch_size):
        test_iterator = \
            tf.data.Dataset.from_tensor_slices(paths).shuffle(buffer_size=100).batch(batch_size=batch_size)
        with tf.device("GPU"):
            y = []
            y_hat = []
            self.lossTracker.result()
            for iter_id, batch_paths in enumerate(test_iterator):
                images, masks, weights = self.get_batch_data(batch_paths=batch_paths, augment=False)
                logits = self.model(images).numpy()
                predicted_labels = np.argmax(logits, axis=-1)
                gt_labels = masks[:, :, :, 0]
                predicted_labels_vec = np.reshape(predicted_labels, newshape=(np.prod(predicted_labels.shape),))
                gt_labels_vec = np.reshape(gt_labels, newshape=(np.prod(gt_labels.shape),))
                weights_vec = np.reshape(weights[:, :, :, 0], newshape=(np.prod(weights[:, :, :, 0].shape),))
                predicted_labels_vec = predicted_labels_vec[weights_vec > 0]
                gt_labels_vec = gt_labels_vec[weights_vec > 0]
                y.append(gt_labels_vec)
                y_hat.append(predicted_labels_vec)
            y = np.concatenate(y)
            y_hat = np.concatenate(y_hat)
        rep = classification_report(y, y_hat)
        print(rep)
        return rep

    def convert_back_to_original_size(self, original_image_shape, result_image):
        curr_ratio = original_image_shape[0] / original_image_shape[1]
        if curr_ratio <= self.inputShape[0] / self.inputShape[1]:
            resize_ratio = self.inputShape[1] / original_image_shape[1]
            new_size = [int(resize_ratio * original_image_shape[0]), self.inputShape[1]]
        else:
            resize_ratio = self.inputShape[0] / original_image_shape[0]
            new_size = [self.inputShape[0], int(resize_ratio * original_image_shape[1])]
        offset_x = self.inputShape[1] - new_size[1]
        offset_y = self.inputShape[0] - new_size[0]
        cropped_result = result_image[offset_y:result_image.shape[0], offset_x:result_image.shape[1]]
        resized_cropped_result = tf.image.resize(np.expand_dims(cropped_result, axis=-1),
                                                 size=(original_image_shape[0], original_image_shape[1]))
        final_image = tf.cast(tf.greater(resized_cropped_result, 0), tf.uint8)
        final_image = final_image.numpy()[:, :, 0]
        return final_image

    def get_result_images(self, visuals_path, paths, batch_size, verbose):
        test_iterator = \
            tf.data.Dataset.from_tensor_slices(paths).shuffle(buffer_size=100).batch(batch_size=batch_size)
        if verbose:
            Utils.create_directory(visuals_path)
        masks_list = []
        with tf.device("GPU"):
            for iter_id, batch_paths in tqdm(enumerate(test_iterator)):
                images, masks, weights = self.get_batch_data(batch_paths=batch_paths, augment=False)
                logits = self.model(images).numpy()
                predicted_labels = np.argmax(logits, axis=-1)
                for s_id, image_path in enumerate(batch_paths):
                    path_str = image_path.numpy().decode("utf-8")
                    root_folder = os.path.split(path_str)[0]
                    file_name = os.path.split(path_str)[1]
                    # Back to the original image
                    original_sized_mask = self.convert_back_to_original_size(
                        original_image_shape=self.originalImagesDict[path_str].shape,
                        result_image=predicted_labels[s_id])
                    assert original_sized_mask.shape[0] == self.originalImagesDict[path_str].shape[0] \
                           and original_sized_mask.shape[1] == self.originalImagesDict[path_str].shape[1]
                    if verbose:
                        original_image_destination_path = os.path.join(visuals_path, file_name)
                        mask_destination_path = os.path.join(visuals_path, file_name[:-4] + "_predicted_mask.png")
                        cv2.imwrite(mask_destination_path, 255 * original_sized_mask)
                        cv2.imwrite(original_image_destination_path,
                                    cv2.cvtColor(self.originalImagesDict[path_str], cv2.COLOR_RGB2BGR))
                    masks_list.append(original_sized_mask)
        return masks_list

    def get_batch_data(self, batch_paths, augment):
        images = []
        masks = []
        weights = []
        for file_path_tf in batch_paths:
            image_resized_padded, dilated_mask_image_resized_padded, weight_mask_padded = \
                self.decode_and_augment_images(file_path=file_path_tf, augment=augment, verbose=False)
            images.append(image_resized_padded)
            masks.append(dilated_mask_image_resized_padded)
            weights.append(weight_mask_padded)
        images = np.stack(images, axis=0)
        masks = np.stack(masks, axis=0)
        weights = np.stack(weights, axis=0)
        return images, masks, weights

    def train(self, paths, epoch_count, batch_size, test_ratio=0.1):
        # Apply train test split
        root_path = os.path.join(self.modelPath, self.modelName)
        Utils.create_directory(path=root_path)

        train_paths, test_paths, = train_test_split(paths, test_size=test_ratio)
        training_images_path = os.path.join(root_path, "training_images.sav")
        test_images_path = os.path.join(root_path, "test_images.sav")
        with open(training_images_path, "wb") as f:
            pickle.dump(train_paths, f)
        with open(test_images_path, "wb") as f:
            pickle.dump(test_paths, f)

        train_iterator = \
            tf.data.Dataset.from_tensor_slices(train_paths).shuffle(buffer_size=100).batch(
                batch_size=batch_size)
        self.imageDict = {}
        self.maskDict = {}

        with tf.device("GPU"):
            for epoch_id in range(epoch_count):
                self.accuracyMetric.reset_states()
                self.lossTracker.result()
                for iter_id, batch_paths in enumerate(train_iterator):
                    t0 = time.time()
                    images, masks, weights = self.get_batch_data(batch_paths=batch_paths, augment=True)
                    t1 = time.time()
                    with tf.GradientTape() as tape:
                        total_loss = self.calculate_loss(images=images, masks=masks, weights=weights)
                    grads = tape.gradient(total_loss, self.model.trainable_variables)
                    t2 = time.time()
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    self.lossTracker.update_state(total_loss)
                    print("Epoch:{0} Iteration:{1} Loss:{2} t1-t0:{3} t2-t1:{4}".format(
                        epoch_id, iter_id,
                        self.lossTracker.result().numpy(),
                        t1 - t0, t2 - t1))
                self.save_model(path=os.path.join(root_path, "model_epoch{0}".format(epoch_id)))
                # training_classification_rep = self.eval(paths=train_paths, batch_size=batch_size)
                test_classification_rep = self.eval(paths=test_paths, batch_size=batch_size)
                profiling_file = open(os.path.join(root_path, 'fit_unet.txt'), 'a')
                profiling_file.write("Epoch:{0} Loss:{1}\n".format(epoch_id, self.lossTracker.result().numpy()))
                profiling_file.write("Training Stats\n")
                # profiling_file.write("{0}\n".format(training_classification_rep))
                profiling_file.write("Test Stats\n")
                profiling_file.write("{0}\n".format(test_classification_rep))
                profiling_file.close()
