import numpy as np
import tensorflow as tf
import os

from data_handling.barcode_dataset import BarcodeDataset
from text_detection.blaze_ssd import BlazeSsdDetector


image_width = 192
image_height = 96
input_dims = (image_height, image_width, 3)
anchor_box_counts = (2, 6)
cluster_count = sum(anchor_box_counts)
batch_size = 64
epoch_count = 1000


def data_processing_for_text_detection():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    file_path = os.path.dirname(__file__)
    barcodes_with_text_detection_path = os.path.join(file_path, "..", "barcodes_with_text_detection")
    barcode_folder_path = os.path.join(file_path, "..", "barcodes")
    model_path = os.path.join(file_path, "..", "saved_models")

    barcode_dataset = BarcodeDataset(dataset_path=barcode_folder_path)
    barcode_dataset.preprocess_dataset(
        barcodes_with_text_detection_path=barcodes_with_text_detection_path, cluster_count=cluster_count)

    blaze_ssd_detector = BlazeSsdDetector(
        model_name="barcode_detector",
        model_path=model_path,
        input_shape=input_dims,
        prior_boxes=barcode_dataset.anchorBoxes
    )
    blaze_ssd_detector.build_detector()
    blaze_ssd_detector.train(batch_size=batch_size, epoch_count=epoch_count,
                             training_set=barcode_dataset.boundingBoxDict)
