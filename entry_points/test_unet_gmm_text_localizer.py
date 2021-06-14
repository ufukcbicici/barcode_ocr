import numpy as np
import tensorflow as tf
import os
import pickle

from anomaly_detection.conv_autoencoder import ConvAE
from data_handling.barcode_dataset import BarcodeDataset
from text_detection.blaze_ssd import BlazeSsdDetector
from text_detection.bounding_box_modelling import BoundingBoxGMMModeller
from text_detection.unet_for_text_detection import Unet
from utils import Utils
import cv2

image_width = 192
image_height = 96
input_dims = (image_height, image_width, 3)
batch_size = 64
epoch_count = 500
unet_model_name = "unet_detector"
layer_width = 2
layer_depth = 1
class_weights = {0: 0.5, 1: 1.0}
l2_lambda = 0.00001
model_epoch = "model_epoch469"
cluster_count = 5
dilation_kernel = 5
gmm_model_name = "gmm_model"
max_connected_component_count = 7


def test_unet_gmm_text_localizer():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    file_path = os.path.dirname(__file__)
    barcodes_with_text_detection_path = os.path.join(file_path, "..", "barcodes_with_text_detection")
    unet_model_path = os.path.join(file_path, "..", "saved_models", unet_model_name)
    training_images_path = os.path.join(unet_model_path, "training_images.sav")
    test_images_path = os.path.join(unet_model_path, "test_images.sav")
    gmm_model_path = os.path.join(os.path.join(file_path, "..", "saved_models"), gmm_model_name)

    with open(training_images_path, "rb") as f:
        train_paths = pickle.load(f)
    with open(test_images_path, "rb") as f:
        test_paths = pickle.load(f)

    unet = Unet(model_name=unet_model_name, model_path=os.path.join(file_path, "..", "saved_models"),
                input_shape=input_dims, layer_width=layer_width, layer_depth=layer_depth, label_count=2,
                dilation_kernel=5, class_weights=class_weights, l2_lambda=l2_lambda)
    unet.load_model(path=os.path.join(unet_model_path, model_epoch))

    bb_gmm_modeller = BoundingBoxGMMModeller.load_gmm_model(output_path=gmm_model_path)

    Utils.create_directory(path=os.path.join(gmm_model_path, "visual_results"))
    Utils.create_directory(path=os.path.join(gmm_model_path, "ground_truth"))
    Utils.create_directory(path=os.path.join(gmm_model_path, "predictions"))
    for path in test_paths:
        image = tf.io.decode_png(tf.io.read_file(path))
        bounding_box, mask_image = bb_gmm_modeller.localize_text_from_image_path(
            path=path, unet=unet,
            max_connected_component_count=max_connected_component_count,
            verbose=False)
        image_name = os.path.split(path)[1]
        image_copy = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        left = int(bounding_box[0])
        top = int(bounding_box[1])
        right = int(bounding_box[2])
        bottom = int(bounding_box[3])
        cv2.rectangle(image_copy, (left, top), (right, bottom), color=(0, 0, 255))
        cv2.rectangle(mask_image, (left, top), (right, bottom), color=(0, 0, 255))
        cv2.imwrite(os.path.join(gmm_model_path, "visual_results", image_name), image_copy)
        cv2.imwrite(os.path.join(gmm_model_path, "visual_results", image_name + "_mask.png"), 255 * mask_image)
        detection_stats_path = os.path.join(barcodes_with_text_detection_path, image_name[:-4] + "_detection_stats.dat")
        with open(detection_stats_path, "rb") as f:
            detections_dict = pickle.load(f)
        gt_bounding_box = np.array(
            [detections_dict["bounding_box"][0, 1],
             detections_dict["bounding_box"][0, 0],
             detections_dict["bounding_box"][1, 1],
             detections_dict["bounding_box"][1, 0]])
        # Ground Truth
        with open(os.path.join(gmm_model_path, "ground_truth", image_name + ".txt"), 'a') as f:
            f.write("{0} {1} {2} {3} {4}\n".format(
                "barcode",
                int(gt_bounding_box[0]),
                int(gt_bounding_box[1]),
                int(gt_bounding_box[2]),
                int(gt_bounding_box[3])))
        # Predictions
        with open(os.path.join(gmm_model_path, "predictions", image_name + ".txt"), 'a') as f:
            f.write("{0} {1} {2} {3} {4} {5}\n".format(
                "barcode",
                1.0,
                left,
                top,
                right,
                bottom))
