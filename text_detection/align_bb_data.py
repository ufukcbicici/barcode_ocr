import tensorflow as tf
import numpy as np

from utils import Utils


class AlignBBData(tf.keras.layers.Layer):
    def __init__(self, prior_box_tensors, **kwargs):
        super().__init__(**kwargs)
        for prior_id, prior_tensor in enumerate(prior_box_tensors):
            self.add_weight("prior_{0}".format(prior_id), shape=prior_tensor.shape,
                            initializer=tf.keras.initializers.Constant(value=0),
                            dtype=prior_tensor.dtype, trainable=False)
            self.weights[-1].assign(prior_tensor)
        assert len(self.weights) == len(prior_box_tensors)
        assert all(["prior_{0}".format(prior_id) in self.weights[prior_id].name
                    for prior_id in range(len(prior_box_tensors))])

    def call(self, inputs, training=None):
        confidence_outputs = inputs[0]
        regression_outputs = inputs[1]
        prior_boxes = self.weights
        # assert confidence_outputs.shape[0] == regression_outputs.shape[0]
        # batch_size = confidence_outputs[0].shape[0]
        batch_size = tf.shape(confidence_outputs[0])[0]

        # Reshape prior boxes: (num_priors, 4)
        prior_boxes_reshaped = [tf.reshape(pb, shape=(pb.shape[0] * pb.shape[1], 4)) for pb in prior_boxes]
        prior_boxes_reshaped = tf.concat(prior_boxes_reshaped, axis=0)
        prior_boxes_reshaped = tf.tile(tf.expand_dims(prior_boxes_reshaped, axis=0), [batch_size, 1, 1])
        prior_count = prior_boxes_reshaped.shape[1]
        # Reshape confidences: (num_samples, num_priors, 1)
        confidence_tensors = []
        confidences_reshaped = []
        for conf in confidence_outputs:
            for idx in range(conf.shape[-1]):
                # confidence_tensors.append(tf.expand_dims(conf[..., idx], axis=-1))
                confidence_tensors.append(tf.expand_dims(conf[:, :, :, idx], axis=-1))

        for conf_arr in confidence_tensors:
            conf_arr_reshaped = tf.reshape(conf_arr,
                                           shape=(-1, conf_arr.shape[1] * conf_arr.shape[2], conf_arr.shape[3]))
            confidences_reshaped.append(conf_arr_reshaped)
        confidences_reshaped = tf.concat(confidences_reshaped, axis=1)

        # Reshape regression parameters: (num_samples, num_priors, 4)
        regression_tensors = []
        regression_reshaped = []
        for reg in regression_outputs:
            for idx in range(reg.shape[-1] // 4):
                # regression_tensors.append(reg[..., 4 * idx:4 * (idx + 1)])
                regression_tensors.append(reg[:, :, :, 4 * idx:4 * (idx + 1)])

        for reg_arr in regression_tensors:
            reg_arr_reshape = tf.reshape(reg_arr, shape=(-1, reg_arr.shape[1] * reg_arr.shape[2], reg_arr.shape[3]))
            regression_reshaped.append(reg_arr_reshape)
        regression_reshaped = tf.concat(regression_reshaped, axis=1)

        # confidences_reshaped:
        # (BatchSize, Sum_n (AnchorCountPerOutput[n] * feature_map_width[n] * feature_map_height[n]), 1)
        # regression_reshaped:
        # (BatchSize, Sum_n (AnchorCountPerOutput[n] * feature_map_width[n] * feature_map_height[n]), 4)
        # prior_boxes_reshaped:
        # (BatchSize, Sum_n (AnchorCountPerOutput[n] * feature_map_width[n] * feature_map_height[n]), 4)
        return confidences_reshaped, regression_reshaped, prior_boxes_reshaped
