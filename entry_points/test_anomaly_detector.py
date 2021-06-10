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
model_epoch = "model_epoch250"


def test_anomaly_detector():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    file_path = os.path.dirname(__file__)
    barcode_model_path = os.path.join(file_path, "..", "saved_models", detector_model_name)

    # Load data related to this model; training and test files; prior boxes
    training_images_path = os.path.join(barcode_model_path, "training_images.sav")
    test_images_path = os.path.join(barcode_model_path, "test_images.sav")

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
    conv_autoencoder.load_model(path=os.path.join(file_path, "..", "saved_models", model_name, model_epoch))
    conv_autoencoder.eval_anomaly_detector(test_paths=test_paths, batch_size=1024, percentile=1.0)

