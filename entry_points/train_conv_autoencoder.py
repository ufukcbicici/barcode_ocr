import numpy as np
import tensorflow as tf
import os
import pickle

from anomaly_detection.conv_autoencoder import ConvAE
from data_handling.barcode_dataset import BarcodeDataset
from text_detection.blaze_ssd import BlazeSsdDetector


patch_width = 16
patch_height = 8
input_dims = (patch_height, patch_width, 3)
batch_size = 64
patch_per_image = 16
epoch_count = 1000
model_name = "patch_autoencoder"
detector_model_name = "barcode_detector_2"
encoder_layers = [(3, 16), (3, 32)]
decoder_layers = [(3, 16), (3, 32)]
do_calculate_mask_obb = False


def train_conv_autoencoder():
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
    if do_calculate_mask_obb:
        barcode_dataset.calculate_mask_oriented_bounding_boxes(barcodes_with_text_detection_path)

    # barcode_dataset.calculate_text_bounding_boxes(
    #     barcodes_with_text_detection_path=barcodes_with_text_detection_path, cluster_count=cluster_count)
    #
    #
    with open(training_images_path, "rb") as f:
        train_paths = pickle.load(f)
    with open(test_images_path, "rb") as f:
        test_paths = pickle.load(f)

    conv_autoencoder = ConvAE(model_name=model_name,
                              model_path=os.path.join(file_path, "..", "saved_models"),
                              original_dim=input_dims,
                              latent_dim=32,
                              layers_encoder=encoder_layers,
                              layers_decoder=decoder_layers)
    conv_autoencoder.train(train_paths=train_paths,
                           test_paths=test_paths,
                           batch_size=batch_size,
                           patch_per_image=patch_per_image,
                           epoch_count=epoch_count)
