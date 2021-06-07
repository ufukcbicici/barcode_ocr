import cv2
import math
import argparse
import os
import numpy as np

from data_handling.barcode_dataset import BarcodeDataset
from data_handling.text_annotator import TextAnnotator
from entry_points.test_ssd_text_detector import test_blaze_ssd_detector
from entry_points.train_ssd_text_detector import train_blaze_ssd_detector
from entry_points.train_conv_autoencoder import train_conv_autoencoder
from text_detection.opencv_pretrained_east_detector import OpencvEastDetector
from utils import Utils

if __name__ == '__main__':
    # train_blaze_ssd_detector()
    # test_blaze_ssd_detector()
    train_conv_autoencoder()
