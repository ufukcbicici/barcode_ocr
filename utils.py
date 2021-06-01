import cv2
import os
import shutil
import numpy as np


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def show_text_detections(image, detections, window_name="Detections", selected_idx=None):
        image_copy = image.copy()
        for detectiod_idx, vertices in enumerate(detections):
            for j in range(4):
                p1 = (int(vertices[j][0]), int(vertices[j][1]))
                p2 = (int(vertices[(j + 1) % 4][0]), int(vertices[(j + 1) % 4][1]))
                if selected_idx is not None and selected_idx == detectiod_idx:
                    cv2.line(image_copy, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.line(image_copy, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(window_name, image_copy)
        # cv2.waitKey(0)

    @staticmethod
    def affine_transform_points_2d(transform_matrix, points):
        assert transform_matrix.shape[0] == transform_matrix.shape[1] == 3
        assert len(points.shape) == 2 and points.shape[1] == 2
        points_homogenous = np.concatenate([points, np.ones_like(points[:, 0])[:, np.newaxis]], axis=1)
        points_transformed = (transform_matrix @ points_homogenous.T).T
        points_transformed = points_transformed[:, 0:2]
        return points_transformed

    @staticmethod
    def show_image_with_normalized_bounding_boxes(image, bb_list, window_name):
        im = image.copy()
        width = image.shape[1]
        height = image.shape[0]
        for bb_arr in bb_list:
            # Comment this out; only for visualizing purposes
            left = int(bb_arr[0] * width)
            top = int(bb_arr[1] * height)
            right = int(bb_arr[2] * width)
            bottom = int(bb_arr[3] * height)
            cv2.rectangle(im, (left, top), (right, bottom), color=(0, 255, 0))
        cv2.imshow(window_name, im)
        cv2.waitKey(0)

    @staticmethod
    def create_directory(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)
