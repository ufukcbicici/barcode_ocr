import cv2
import numpy as np
import pickle
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from collections import Counter
from sklearn_extra.cluster import KMedoids

from utils import Utils


class BarcodeDataset:
    def __init__(self, dataset_path):
        self.datasetPath = dataset_path
        self.allImagePaths = [os.path.join(self.datasetPath, f)
                              for f in listdir(self.datasetPath) if isfile(join(self.datasetPath, f))]
        self.anchorBoxes = None
        self.boundingBoxDict = None

    @staticmethod
    def cluster_bounding_boxes(bb_coords, cluster_count):
        kmedoids = KMedoids(n_clusters=cluster_count)
        widths = bb_coords[:, 2] - bb_coords[:, 0]
        heights = bb_coords[:, 3] - bb_coords[:, 1]
        bb_dimensions = np.stack([widths, heights], axis=1)
        kmedoids.fit(X=bb_dimensions)
        counter = Counter(kmedoids.labels_)
        print("counter={0}".format(counter))
        return kmedoids.cluster_centers_

    def preprocess_dataset(self, barcodes_with_text_detection_path, cluster_count):
        all_paths = [os.path.join(barcodes_with_text_detection_path, f)
                     for f in listdir(barcodes_with_text_detection_path)
                     if isfile(join(barcodes_with_text_detection_path, f))]
        assert len(all_paths) % 4 == 0
        # All barcode image files and their corresponding data
        barcode_file_path_dict = {}
        barcode_data_path_dict = {}

        for path in all_paths:
            file_name = os.path.split(path)[1]
            if ".png" in file_name and "_mask" not in file_name and "_bb" not in file_name:
                barcode_file_path_dict[file_name[:-4]] = path
            if ".dat" in file_name:
                barcode_data_path_dict[file_name[:file_name.index("_")]] = path
        files_set1 = set(barcode_file_path_dict.keys())
        files_set2 = set(barcode_data_path_dict.keys())
        assert files_set1 == files_set2
        # All barcode image sizes and bounding box list for texts
        barcode_image_sizes_dict = {}
        barcode_text_bbs_dict = {}
        for file_name in tqdm(barcode_file_path_dict):
            image = cv2.imread(barcode_file_path_dict[file_name])
            data_path = barcode_data_path_dict[file_name]
            with open(data_path, "rb") as f:
                data_dict = pickle.load(f)
            # "bounding_box": np.stack([scaled_top_left, scaled_bottom_right], axis=0)
            bounding_box = data_dict["bounding_box"]
            bounding_box_normalized = np.clip(
                np.array(
                    [bounding_box[0, 1] / image.shape[1],
                     bounding_box[0, 0] / image.shape[0],
                     bounding_box[1, 1] / image.shape[1],
                     bounding_box[1, 0] / image.shape[0]]), 0.0, 1.0)
            # Utils.show_image_with_normalized_bounding_boxes(image, [bounding_box_normalized], "normalized_bb")
            barcode_image_sizes_dict[file_name] = np.array([image.shape[0], image.shape[1]])
            barcode_text_bbs_dict[file_name] = bounding_box_normalized
        # Analyze image sizes
        image_sizes = np.stack([shp for shp in barcode_image_sizes_dict.values()], axis=0)
        height_width_ratios = np.array([shp[0] / shp[1] for shp in image_sizes])
        assert all([image_sizes[i, 0] / image_sizes[i, 1] == height_width_ratios[i]
                    for i in range(image_sizes.shape[0])])
        mean_size = np.mean(image_sizes, axis=0)
        print("Mean Size:{0}".format(mean_size))
        print("Min Ratio:{0}".format(np.min(height_width_ratios)))
        print("Max Ratio:{0}".format(np.max(height_width_ratios)))
        # Cluster bounding boxes
        bb_coords = np.stack([bb for bb in barcode_text_bbs_dict.values()], axis=0)
        self.anchorBoxes = BarcodeDataset.cluster_bounding_boxes(bb_coords=bb_coords, cluster_count=cluster_count)
        self.boundingBoxDict = {}
        for k, v in barcode_text_bbs_dict.items():
            self.boundingBoxDict[os.path.join(barcodes_with_text_detection_path, "{0}.png".format(k))] = [v]
