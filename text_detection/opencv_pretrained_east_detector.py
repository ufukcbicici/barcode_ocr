import cv2
import math
import numpy as np


class OpencvEastDetector:
    def __init__(self, model_path, conf_threshold, nms_threshold, input_width, input_height):
        self.modelPath = model_path
        self.confThreshold = conf_threshold
        self.nmsThreshold = nms_threshold
        self.inpWidth = input_width
        self.inpHeight = input_height
        self.net = cv2.dnn.readNet(self.modelPath)
        self.outputLayers = []
        self.outputLayers.append("feature_fusion/Conv_7/Sigmoid")
        self.outputLayers.append("feature_fusion/concat_3")

    def decode(self, scores, geometry, score_thresh):
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scores_data = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            angles_data = geometry[0][4][y]
            for x in range(0, width):
                score = scores_data[x]

                # If score is lower than threshold score, move to next x
                if score < score_thresh:
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = angles_data[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = (
                    [offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]

    def get_detections(self, image, filter=True):
        self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

        height_ = image.shape[0]
        width_ = image.shape[1]
        rW = width_ / float(self.inpWidth)
        rH = height_ / float(self.inpHeight)
        blob = cv2.dnn.blobFromImage(image, 1.0, (self.inpWidth, self.inpHeight), (123.68, 116.78, 103.94), True, False)
        self.net.setInput(blob)
        output = self.net.forward(self.outputLayers)
        scores = output[0]
        geometry = output[1]
        [boxes, confidences] = self.decode(scores, geometry, self.confThreshold)
        # Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.confThreshold, self.nmsThreshold)
        detections = []
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            detections.append(vertices)
        if filter:
            selected_detection_idx = self.filter_detections_with_heuristics(image=image, detections=detections)
            return detections[selected_detection_idx]
        return detections

    def filter_detections_with_heuristics(self, image, detections):
        # Always pick the detections which are nearest to the bottom right corner of the image
        detection_distances = []
        image_bottom_right = np.array([image.shape[1]-1, image.shape[0]-1])
        for vertices in detections:
            distances = np.linalg.norm(vertices - image_bottom_right[np.newaxis, :], axis=1)
            detection_distances.append(np.min(distances))
        selected_detection_idx = np.argmin(np.array(detection_distances))
        return selected_detection_idx
