import numpy as np
import tensorflow as tf
import os
import pickle

from anomaly_detection.conv_autoencoder import ConvAE
from data_handling.barcode_dataset import BarcodeDataset
from text_detection.blaze_ssd import BlazeSsdDetector
from text_detection.bounding_box_modelling import BoundingBoxGMMModeller
from text_detection.unet_for_text_detection import Unet
from text_recognition.cnn_rnn_ctc_model import CnnRnnCtcModel

image_height = 32
batch_size = 64
epoch_count = 500
model_name = "cnn_rnn_ctc_reader"
l2_lambda = 0.00001


def train_cnn_rnn_ctc_model():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    file_path = os.path.dirname(__file__)
    barcodes_with_text_detection_path = os.path.join(file_path, "..", "barcodes_with_text_detection")
    training_images_path = os.path.join(file_path, "..", "train_test_files", "training_images.sav")
    test_images_path = os.path.join(file_path, "..", "train_test_files", "test_images.sav")
    # model_path = os.path.join(file_path, "..", "saved_models", model_name)

    with open(training_images_path, "rb") as f:
        train_paths = pickle.load(f)
    with open(test_images_path, "rb") as f:
        test_paths = pickle.load(f)

    cnn_rnn_ctc_model = CnnRnnCtcModel(
        model_path=os.path.join(file_path, "..", "saved_models"),
        model_name=model_name,
        input_height=image_height, l2_lambda=l2_lambda)
    cnn_rnn_ctc_model.train(train_paths=train_paths, test_paths=test_paths,
                            batch_size=batch_size, epoch_count=epoch_count)
