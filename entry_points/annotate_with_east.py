import os
import cv2
import numpy as np

from data_handling.barcode_dataset import BarcodeDataset
from data_handling.text_annotator import TextAnnotator
from text_detection.opencv_pretrained_east_detector import OpencvEastDetector


def annotate_with_east():
    file_path = os.path.dirname(__file__)
    barcode_folder_path = os.path.join(file_path, "..", barcodes)
    barcode_image_path = os.path.join(barcode_folder_path, "FCZ2510R7AM.png")
    image = cv2.imread(barcode_image_path)

    model_path = os.path.join(file_path, "models", "frozen_east_text_detection.pb")
    confThreshold = 0.5
    nmsThreshold = 0.4
    inpWidth = 320
    inpHeight = 320

    barcode_dataset = BarcodeDataset(dataset_path=barcode_folder_path)
    random_image_count = 10
    selected_image_paths = np.random.choice(barcode_dataset.allImagePaths, random_image_count, False)

    opencv_east_detector = OpencvEastDetector(model_path=model_path,
                                              conf_threshold=confThreshold,
                                              nms_threshold=nmsThreshold,
                                              input_width=inpWidth,
                                              input_height=inpHeight)

    unprocessed_files_path = os.path.join(file_path, "barcodes_unprocessed")
    processed_files_path = os.path.join(file_path, "barcodes_with_text_detection")
    barcodes_with_problematic_text_detection = os.path.join(file_path, "barcodes_with_problematic_text_detection")

    TextAnnotator.annotate_files(text_detector=opencv_east_detector,
                                 unprocessed_files_path=unprocessed_files_path,
                                 processed_files_path=processed_files_path,
                                 problematic_files_path=barcodes_with_problematic_text_detection,
                                 bb_scale_ratio=1.05)
