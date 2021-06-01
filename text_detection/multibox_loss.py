import tensorflow as tf
import numpy as np

from utils import Utils


class MultiboxLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def calculate(self, confidence_outputs, regression_outputs, prior_boxes, ground_truths):
    #     # Match bounding boxes with

    def match_priors_with_ground_truth(self, priors, gt_box):
        h = priors.shape[0]
        w = priors.shape[1]
        gt_tensor = tf.tile(tf.expand_dims(tf.expand_dims(gt_box, axis=0), axis=0), [h, w, 1])
        assert priors.shape == gt_tensor.shape
        union_tensor = tf.stack([priors, gt_tensor], axis=0)

        # Intersection area calculation in a vectorized way
        left_top_tensor = union_tensor[..., :2]
        right_bottom_tensor = union_tensor[..., 2:]
        intersection_left_top = tf.reduce_max(left_top_tensor, axis=0)
        intersection_right_bottom = tf.reduce_min(right_bottom_tensor, axis=0)
        intersection_widths = intersection_right_bottom[..., 0] - intersection_left_top[..., 0]
        intersection_heights = intersection_right_bottom[..., 1] - intersection_left_top[..., 1]
        intersection_areas = intersection_widths * intersection_heights

        # Union area calculation in a vectorized way
        prior_widths = priors[..., 2] - priors[..., 0]
        prior_heights = priors[..., 3] - priors[..., 1]
        prior_areas = prior_widths * prior_heights

        # Union area calculation in a vectorized way
        gt_widths = gt_tensor[..., 2] - gt_tensor[..., 0]
        gt_heights = gt_tensor[..., 3] - gt_tensor[..., 1]
        gt_areas = gt_widths * gt_heights

        union_areas = (prior_areas + gt_areas) - intersection_areas
        iou_tensor = intersection_areas / union_areas

        # TODO: Convert this into a unit test later. Can comment this out for the production code.
        priors_np_arr = priors.numpy()
        gt_box_np_arr = gt_box.numpy()
        iou_tensor_np_arr = np.zeros(shape=(priors_np_arr.shape[0], priors_np_arr.shape[1]))
        for i in range(priors_np_arr.shape[0]):
            for j in range(priors_np_arr.shape[1]):
                iou_tensor_np_arr[i, j] = Utils.get_iou_of_two_rects(priors_np_arr[i, j], gt_box_np_arr)
        assert np.allclose(iou_tensor.numpy(), iou_tensor_np_arr)

        return iou_tensor

    def call(self, inputs, training=None):
        confidences_reshaped = inputs[0]
        regression_reshaped = inputs[1]
        prior_boxes_reshaped = inputs[2]
        ground_truths_per_image = inputs[3]
        batch_size = len(ground_truths_per_image)
        prior_count = prior_boxes_reshaped.shape[1]

        # # Reshape prior boxes: (num_priors, 4)
        # prior_boxes_reshaped = [tf.reshape(pb, shape=(pb.shape[0] * pb.shape[1], 4)) for pb in prior_boxes]
        # prior_boxes_reshaped = tf.concat(prior_boxes_reshaped, axis=0)
        # prior_boxes_reshaped = tf.tile(tf.expand_dims(prior_boxes_reshaped, axis=0), [batch_size, 1, 1])
        # prior_count = prior_boxes_reshaped.shape[1]
        # # Reshape confidences: (num_samples, num_priors, 1)
        # confidence_tensors = []
        # confidences_reshaped = []
        # for conf in confidence_outputs:
        #     for idx in range(conf.shape[-1]):
        #         confidence_tensors.append(tf.expand_dims(conf[..., idx], axis=-1))
        #
        # for conf_arr in confidence_tensors:
        #     confidences_reshaped.append(
        #         tf.reshape(conf_arr,
        #                    shape=(conf_arr.shape[0], conf_arr.shape[1] * conf_arr.shape[2], conf_arr.shape[3])))
        # confidences_reshaped = tf.concat(confidences_reshaped, axis=1)
        #
        # # Reshape regression parameters: (num_samples, num_priors, 4)
        # regression_tensors = []
        # regression_reshaped = []
        # for reg in regression_outputs:
        #     for idx in range(reg.shape[-1] // 4):
        #         regression_tensors.append(reg[..., 4 * idx:4 * (idx + 1)])
        #
        # for reg_arr in regression_tensors:
        #     regression_reshaped.append(
        #         tf.reshape(reg_arr,
        #                    shape=(reg_arr.shape[0], reg_arr.shape[1] * reg_arr.shape[2], reg_arr.shape[3])))
        # regression_reshaped = tf.concat(regression_reshaped, axis=1)

        # Reshape ground truths: (num_samples, max_num_of_gt_boxes_in_samples, 4)
        max_num_of_gt_boxes_in_samples = max([len(bb_list) for bb_list in ground_truths_per_image])
        # Duplicate one of the gt boxes for images with #gt_box < max_num_of_gt_boxes_in_samples
        gt_list_expanded = []
        for bb_list in ground_truths_per_image:
            bb_expanded = []
            bb_expanded.extend(bb_list)
            dummy_bbox = -1.0 * tf.ones_like(bb_list[0])
            bb_expanded.extend([dummy_bbox] * (max_num_of_gt_boxes_in_samples - len(bb_list)))
            bb_arr = tf.stack(bb_expanded, axis=0)
            gt_list_expanded.append(bb_arr)
        gt_tensor = tf.stack(gt_list_expanded, axis=0)

        # Measure IoU metric between each prior and each ground truth of each image in the minibatch.
        iou_tensors = []
        for gt_index in range(gt_tensor.shape[1]):
            gt = tf.expand_dims(gt_tensor[:, gt_index], axis=1)
            gt = tf.tile(gt, [1, prior_count, 1])
            union_tensor = tf.stack([prior_boxes_reshaped, gt], axis=1)

            # Intersection area calculation in a vectorized way
            left_top_tensor = union_tensor[..., :2]
            right_bottom_tensor = union_tensor[..., 2:]
            intersection_left_top = tf.reduce_max(left_top_tensor, axis=1)
            intersection_right_bottom = tf.reduce_min(right_bottom_tensor, axis=1)
            intersection_widths = intersection_right_bottom[..., 0] - intersection_left_top[..., 0]
            intersection_heights = intersection_right_bottom[..., 1] - intersection_left_top[..., 1]
            intersection_widths = tf.clip_by_value(intersection_widths, clip_value_min=0.0, clip_value_max=100.0)
            intersection_heights = tf.clip_by_value(intersection_heights, clip_value_min=0.0, clip_value_max=100.0)
            intersection_areas = intersection_widths * intersection_heights
            #
            # Union area calculation in a vectorized way
            prior_widths = prior_boxes_reshaped[..., 2] - prior_boxes_reshaped[..., 0]
            prior_heights = prior_boxes_reshaped[..., 3] - prior_boxes_reshaped[..., 1]
            prior_areas = prior_widths * prior_heights
            #
            # Union area calculation in a vectorized way
            gt_widths = gt[..., 2] - gt[..., 0]
            gt_heights = gt[..., 3] - gt[..., 1]
            gt_areas = gt_widths * gt_heights
            #
            union_areas = (prior_areas + gt_areas) - intersection_areas
            iou_tensor = intersection_areas / union_areas
            iou_tensors.append(iou_tensor)

            # TODO: This is for debugging purposes. Can be isolated as a unit test later. Comment out in the production.
            # iou_tensor_np = np.zeros(shape=(batch_size, prior_count))
            # for sample_id in range(batch_size):
            #     gt_box = gt_tensor[sample_id, gt_index].numpy()
            #     for prior_id in range(prior_count):
            #         prior_box = prior_boxes_reshaped[sample_id, prior_id].numpy()
            #         iou_tensor_np[sample_id, prior_id] = Utils.get_iou_of_two_rects(rect1=gt_box, rect2=prior_box)
            # assert np.allclose(iou_tensor.numpy(), iou_tensor_np)
            # assert 0.0 <= iou_tensor.numpy().min() and iou_tensor.numpy().max() <= 1.0
        iou_tensors = tf.stack(iou_tensors, axis=-1)

        # Match priors with ground truths
        # SSD Paper:
        # 1) We begin by matching each ground truth box to the default box with the best
        # jaccard overlap (as in MultiBox [7]).
        # 2) Unlike MultiBox, we then match default boxes to any ground truth with jaccard overlap
        # higher than a threshold (0.5).

        prior_match_map = []
        # Step 1: Assign each ground truth box to the best matching prior box.
        gt_to_prior_matches = tf.argmax(iou_tensors, axis=1)
        gt_to_prior_best_scores = tf.reduce_max(iou_tensors, axis=1)

        # Step 2: Assign each prior box to ground truths, if IoU is over 0.5
        for gt_index in range(gt_to_prior_matches.shape[1]):
            # The "1" entry will correspond to the best gt -> prior match
            match_arr_gt_to_prior = tf.cast(tf.one_hot(gt_to_prior_matches[:, gt_index], prior_count), tf.bool)
            # If gt_to_prior_best_scores == 0, then this corresponds to a match with a dummy ground truth.
            # We need to set these to 0.
            match_with_dummy_arr = tf.expand_dims(gt_to_prior_best_scores[:, gt_index] > 0.0, axis=-1)
            match_arr_gt_to_prior = tf.logical_and(match_arr_gt_to_prior, match_with_dummy_arr)
            # The "1" entries will correspond to IoU(gt,prior) > 0.5
            iou_arr = iou_tensors[:, :, gt_index]
            match_arr_prior_to_gt = tf.greater_equal(iou_arr, 0.5)
            # Now, combine both arrays with logical or.
            match_arr = tf.logical_or(match_arr_gt_to_prior, match_arr_prior_to_gt)
            prior_match_map.append(match_arr)
        prior_match_map = tf.stack(prior_match_map, axis=-1)

        # Now, we have matched every prior to appropriate ground truths. Now; we are going to do the hard negative
        # mining first; then we will calculate the loss.

        # First; we are going to calculate the localization loss.
        localization_losses = []
        confidence_losses = []
        hard_negative_ratio = 3
        alpha = 1.0
        total_positives = []
        for gt_index in range(gt_tensor.shape[1]):
            gt = tf.expand_dims(gt_tensor[:, gt_index], axis=1)
            # Match map
            match_map = prior_match_map[..., gt_index]
            # All ground truth boxes, tiled to match all priors
            gt = tf.tile(gt, [1, prior_count, 1])
            # Ground Truths - Convert to center-size coordinates
            gt_w = gt[..., 2] - gt[..., 0]
            gt_h = gt[..., 3] - gt[..., 1]
            gt_cx = gt[..., 0] + 0.5 * gt_w
            gt_cy = gt[..., 1] + 0.5 * gt_h
            gt_center_coords = tf.stack([gt_cx, gt_cy, gt_w, gt_h], axis=-1)
            # Priors - Convert to center-size coodinates
            priors_w = prior_boxes_reshaped[..., 2] - prior_boxes_reshaped[..., 0]
            priors_h = prior_boxes_reshaped[..., 3] - prior_boxes_reshaped[..., 1]
            priors_cx = prior_boxes_reshaped[..., 0] + 0.5 * priors_w
            priors_cy = prior_boxes_reshaped[..., 1] + 0.5 * priors_h
            prior_center_coords = tf.stack([priors_cx, priors_cy, priors_w, priors_h], axis=-1)
            # Calculate regression targets - Smooth L1 Loss
            y_cx = (gt_cx - priors_cx) / priors_w
            y_cy = (gt_cy - priors_cy) / priors_h
            y_w = tf.math.log(gt_w / priors_w)
            y_h = tf.math.log(gt_h / priors_h)
            y_pred = tf.stack([y_cx, y_cy, y_w, y_h], axis=-1)
            regression_losses = tf.keras.losses.huber(y_pred, regression_reshaped)
            # Only let through gt matches
            masked_regression_losses = tf.cast(match_map, tf.float32) * regression_losses
            localization_losses.append(masked_regression_losses)

            # Calculate confidence losses, with the hard negative ratio
            labels = tf.expand_dims(tf.cast(match_map, tf.int32), axis=-1)
            classification_losses = tf.losses.binary_crossentropy(labels, confidences_reshaped, from_logits=True)

            # Hard negative mining
            num_pos = tf.reduce_sum(tf.cast(match_map, tf.int32))
            total_positives.append(num_pos)
            num_neg = hard_negative_ratio * num_pos
            # Set the losses for positive samples to negative. Such that they do not contribute to sorting
            # for the hard negative finding.
            positive_mask = tf.where(match_map, -1.0, 1.0)
            classification_losses_positives_masked = positive_mask * classification_losses
            losses_flat = tf.reshape(classification_losses_positives_masked,
                                     shape=(classification_losses_positives_masked.shape[0] *
                                            classification_losses_positives_masked.shape[1],))
            losses_sorted = tf.sort(losses_flat, direction='DESCENDING')
            threshold_value = losses_sorted[num_neg - 1]
            # Now; calculate a mask, where all entries larger than "threshold_value" are set to 1, others 0.
            # Positive samples won't contribute since we have already negated them.
            negative_mask = tf.greater_equal(classification_losses_positives_masked, threshold_value)
            # The final mask is the logical or of two masks: One for the positives, one for the negatives.
            confidence_mask = tf.logical_or(tf.cast(match_map, tf.bool), negative_mask)
            masked_classification_losses = tf.cast(confidence_mask, tf.float32) * classification_losses
            confidence_losses.append(masked_classification_losses)

        total_positive_count = tf.add_n(total_positives)
        loss_loc = tf.stack(localization_losses, axis=-1)
        loss_conf = tf.stack(confidence_losses, axis=-1)
        loss_loc = tf.reduce_sum(loss_loc)
        loss_conf = tf.reduce_sum(loss_conf)
        loss_loc = (1.0 / tf.cast(total_positive_count, tf.float32)) * loss_loc
        loss_conf = (1.0 / tf.cast(total_positive_count, tf.float32)) * loss_conf
        total_loss = loss_conf + alpha*loss_loc
        return total_loss

