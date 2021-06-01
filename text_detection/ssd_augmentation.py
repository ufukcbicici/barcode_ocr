import tensorflow as tf
import numpy as np
import cv2

from utils import Utils


class SsdAugmentation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.brightness_max_delta = kwargs["brightness_max_delta"]
        self.contrast_range = kwargs["contrast_range"]
        self.hue_max_delta = kwargs["hue_max_delta"]
        self.saturation_range = kwargs["saturation_range"]

    # super(RandomColorDistortion, self).__init__(**kwargs)
    # self.contrast_range = contrast_range
    # self.brightness_delta = brightness_delta

    def flip_bounding_box_vertically(self, bb):
        left = bb[0]
        right = bb[2]
        x0 = left + 2.0 * (0.5 - left)
        x1 = right + 2.0 * (0.5 - right)
        new_left = min(x0, x1)
        new_right = max(x0, x1)
        bb_arr = tf.concat([new_left, bb[1], new_right, bb[3]], axis=0)
        return bb_arr

    def call(self, inputs, training=None):
        # TODO: Implement this
        return None

        # if not training:
        #     return inputs
        # images = inputs[0]
        # bounding_boxes = inputs[1]
        #
        # img_count = len(images)
        # transformations = tf.random.uniform(
        #     (img_count,), minval=0, maxval=2, dtype=tf.dtypes.int32, seed=None, name=None
        # )
        #
        # images_augmented = []
        # bounding_boxes_augmented = []
        # for idx in range(img_count):
        #     # For visualization
        #     # im = images[idx].numpy()
        #     # Utils.show_image_with_normalized_bounding_boxes(image=im,
        #     #                                                 bb_list=[b.numpy() for b in bounding_boxes[idx]],
        #     #                                                 window_name="normal_image")
        #     # Do nothing
        #     if transformations[idx] == 0:
        #         images_augmented.append(images[idx])
        #         bounding_boxes_augmented.append(bounding_boxes[idx])
        #     # Flip vertically
        #     elif transformations[idx] == 1:
        #         augmented_image = tf.image.flip_left_right(images[idx])
        #         images_augmented.append(augmented_image)
        #         # Flip bounding boxes
        #         augmented_bb_list = []
        #         for bb in bounding_boxes[idx]:
        #             augmented_bb = self.flip_bounding_box_vertically(bb)
        #             augmented_bb_list.append(augmented_bb)
        #         bounding_boxes_augmented.append(augmented_bb_list)
        #     else:
        #         raise ValueError("Not suitable transformation.")
        #     # For visualization
        #     # im = images_augmented[idx].numpy()
        #     # Utils.show_image_with_normalized_bounding_boxes(
        #     #     image=im,
        #     #     bb_list=[b.numpy() for b in bounding_boxes_augmented[idx]],
        #     #     window_name="augmented_image")
        # return images_augmented, bounding_boxes_augmented

        # contrast = np.random.uniform(
        #     self.contrast_range[0], self.contrast_range[1])
        # brightness = np.random.uniform(
        #     self.brightness_delta[0], self.brightness_delta[1])
        #
        # images = tf.image.adjust_contrast(images, contrast)
        # images = tf.image.adjust_brightness(images, brightness)
        # images = tf.clip_by_value(images, 0, 1)
        # return images
