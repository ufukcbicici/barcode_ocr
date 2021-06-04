import numpy as np
import tensorflow as tf
import os
import pickle

from data_handling.barcode_dataset import BarcodeDataset
from text_detection.blaze_ssd import BlazeSsdDetector


image_width = 192
image_height = 96
input_dims = (image_height, image_width, 3)
anchor_box_counts = (2, 6)
cluster_count = sum(anchor_box_counts)
batch_size = 64
epoch_count = 1000
model_name = "barcode_detector"
model_epoch = "model_epoch200"


def test_blaze_ssd_detector():
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    file_path = os.path.dirname(__file__)
    barcodes_with_text_detection_path = os.path.join(file_path, "..", "barcodes_with_text_detection")
    barcode_folder_path = os.path.join(file_path, "..", "barcodes")
    model_path = os.path.join(file_path, "..", "saved_models")
    root_path = os.path.join(model_path, model_name)

    # Load data related to this model; training and test files; prior boxes
    training_images_path = os.path.join(root_path, "training_images.sav")
    test_images_path = os.path.join(root_path, "test_images.sav")
    prior_boxes_path = os.path.join(root_path, "prior_boxes.sav")

    with open(training_images_path, "rb") as f:
        train_paths = pickle.load(f)
    with open(test_images_path, "rb") as f:
        test_paths = pickle.load(f)
    with open(prior_boxes_path, "rb") as f:
        prior_boxes = pickle.load(f)

    barcode_dataset = BarcodeDataset(dataset_path=barcode_folder_path)
    barcode_dataset.preprocess_dataset(
        barcodes_with_text_detection_path=barcodes_with_text_detection_path, cluster_count=cluster_count)

    blaze_ssd_detector = BlazeSsdDetector(
        model_name=model_name,
        model_path=model_path,
        input_shape=input_dims,
        prior_boxes=prior_boxes
    )
    blaze_ssd_detector.build_detector()
    blaze_ssd_detector.load_model(path=os.path.join(model_path, model_name, model_epoch))
    visual_results_path = os.path.join(root_path, "visual_results")
    txt_results_path = os.path.join(root_path, "txt_results")

    blaze_ssd_detector.eval_images(
        training_set=barcode_dataset.boundingBoxDict,
        image_paths=test_paths,
        batch_size=batch_size,
        visual_results_path=visual_results_path,
        txt_results_path=txt_results_path,
        positive_threshold=0.5,
        iou_threshold=0.1)
    print("X")

    #
    # barcode_dataset = BarcodeDataset(dataset_path=barcode_folder_path)
    # barcode_dataset.preprocess_dataset(
    #     barcodes_with_text_detection_path=barcodes_with_text_detection_path, cluster_count=cluster_count)
    #
    # blaze_ssd_detector = BlazeSsdDetector(
    #     model_name="barcode_detector",
    #     model_path=model_path,
    #     input_shape=input_dims,
    #     prior_boxes=barcode_dataset.anchorBoxes
    # )
    # blaze_ssd_detector.build_detector()
    # blaze_ssd_detector.train(batch_size=batch_size, epoch_count=epoch_count,
    #                          training_set=barcode_dataset.boundingBoxDict)
