import numpy as np
import tensorflow as tf
import os
import pickle

from anomaly_detection.conv_autoencoder import ConvAE
from data_handling.barcode_dataset import BarcodeDataset
from text_detection.blaze_ssd import BlazeSsdDetector
from text_detection.unet_for_text_detection import Unet

image_width = 192
image_height = 96
input_dims = (image_height, image_width, 3)
batch_size = 64
epoch_count = 500
model_name = "unet_detector"
detector_model_name = "barcode_detector_2"
layer_width = 2
layer_depth = 1
class_weights = {0: 0.5, 1: 1.0}
l2_lambda = 0.00001


def train_unet_detector():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    file_path = os.path.dirname(__file__)
    barcodes_with_text_detection_path = os.path.join(file_path, "..", "barcodes_with_text_detection")
    barcode_folder_path = os.path.join(file_path, "..", "barcodes")
    barcode_model_path = os.path.join(file_path, "..", "saved_models", detector_model_name)

    barcode_dataset = BarcodeDataset(dataset_path=barcode_folder_path)
    # Load data related to this model; training and test files; prior boxes
    training_images_path = os.path.join(barcode_model_path, "training_images.sav")
    test_images_path = os.path.join(barcode_model_path, "test_images.sav")
    prior_boxes_path = os.path.join(barcode_model_path, "prior_boxes.sav")

    barcode_dataset = BarcodeDataset(dataset_path=barcode_folder_path)
    barcode_dataset.calculate_text_bounding_boxes(barcodes_with_text_detection_path=barcodes_with_text_detection_path)

    paths = []
    for path in barcode_dataset.boundingBoxDict.keys():
        paths.append(path)

    unet = Unet(model_name=model_name, model_path=os.path.join(file_path, "..", "saved_models"),
                input_shape=input_dims, layer_width=layer_width, layer_depth=layer_depth, label_count=2,
                dilation_kernel=5, class_weights=class_weights, l2_lambda=l2_lambda)
    unet.build_network()
    unet.train(paths=paths, batch_size=batch_size, epoch_count=epoch_count)
