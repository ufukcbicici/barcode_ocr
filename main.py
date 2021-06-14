import cv2
import math
import argparse
import os
import numpy as np

from data_handling.barcode_dataset import BarcodeDataset
from data_handling.text_annotator import TextAnnotator
from entry_points.test_anomaly_detector import test_anomaly_detector
from entry_points.test_ssd_text_detector import test_blaze_ssd_detector
from entry_points.test_unet_detector import test_unet_detector
from entry_points.test_unet_gmm_text_localizer import test_unet_gmm_text_localizer
from entry_points.train_anomaly_detector import train_anomaly_detector
from entry_points.train_gmm_bb_model import train_gmm_bb_likelihood
from entry_points.train_ssd_text_detector import train_blaze_ssd_detector
from entry_points.train_conv_autoencoder import train_conv_autoencoder
from entry_points.train_unet_detector import train_unet_detector
from text_detection.opencv_pretrained_east_detector import OpencvEastDetector
from utils import Utils
from sklearn_extra.cluster import KMedoids

if __name__ == '__main__':
    # train_blaze_ssd_detector()
    # test_blaze_ssd_detector()
    # train_conv_autoencoder()
    # train_anomaly_detector()
    # test_anomaly_detector()
    # train_unet_detector()
    # test_unet_detector()
    # train_gmm_bb_likelihood()
    test_unet_gmm_text_localizer()