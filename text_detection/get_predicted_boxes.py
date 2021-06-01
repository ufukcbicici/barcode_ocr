import tensorflow as tf
import numpy as np

from utils import Utils


class GetPredictedBoxes(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        predicted_offsets = inputs[0]
        priors = inputs[1]

        # Tflite does not support strided slices.
        # g_cx = predicted_offsets[..., 0]
        # g_cy = predicted_offsets[..., 1]
        # g_w = predicted_offsets[..., 2]
        # g_h = predicted_offsets[..., 3]
        # priors_w = priors[..., 2] - priors[..., 0]
        # priors_h = priors[..., 3] - priors[..., 1]
        # priors_cx = priors[..., 0] + 0.5 * priors_w
        # priors_cy = priors[..., 1] + 0.5 * priors_h

        g_cx = predicted_offsets[:, :, 0]
        g_cy = predicted_offsets[:, :, 1]
        g_w = predicted_offsets[:, :, 2]
        g_h = predicted_offsets[:, :, 3]
        priors_w = priors[:, :, 2] - priors[:, :, 0]
        priors_h = priors[:, :, 3] - priors[:, :, 1]
        priors_cx = priors[:, :, 0] + 0.5 * priors_w
        priors_cy = priors[:, :, 1] + 0.5 * priors_h

        diff_cx = g_cx * priors_w
        cx_hat = priors_cx + diff_cx
        diff_cy = g_cy * priors_h
        cy_hat = priors_cy + diff_cy
        cw_hat = tf.exp(g_w) * priors_w
        ch_hat = tf.exp(g_h) * priors_h

        c_left_hat = cx_hat - 0.5 * cw_hat
        c_top_hat = cy_hat - 0.5 * ch_hat
        c_right_hat = cx_hat + 0.5 * cw_hat
        c_bottom_hat = cy_hat + 0.5 * ch_hat

        predicted_boxes = tf.stack([c_left_hat, c_top_hat, c_right_hat, c_bottom_hat], axis=-1)
        return predicted_boxes
