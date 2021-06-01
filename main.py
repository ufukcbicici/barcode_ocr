import cv2
import math
import argparse
import os
import numpy as np

from data_handling.barcode_dataset import BarcodeDataset
from data_handling.text_annotator import TextAnnotator
from entry_points.text_detector_training import data_processing_for_text_detection
from text_detection.opencv_pretrained_east_detector import OpencvEastDetector
from utils import Utils

if __name__ == '__main__':
    data_processing_for_text_detection()
