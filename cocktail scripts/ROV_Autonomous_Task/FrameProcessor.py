import cv2
import numpy as np
import sys
from DetectionPreprocessing import *

class FramePreprocessor(DetectionPreprocessor):
    """
    Class for preprocessing a frame with a given color range and blur.
    """

    def __init__(self, frame=None, l_b=[0, 50, 50],
                 u_b=[10, 255, 255],
                 l_b2=[170, 50, 50],
                 u_b2=[180, 255, 255],
                 blur=(5, 5)):
        """
        Initialize the FramePreprocessor class.

        Parameters:
        - frame (numpy.ndarray): Input frame.
        - l_b (list): Lower bound for color range.
        - u_b (list): Upper bound for color range.
        - l_b2 (list): Lower bound for color range 2.
        - u_b2 (list): Upper bound for color range 2.
        - blur (tuple): Size of the blur kernel.
        """
        self.frame = frame
        self.l_b = np.array(l_b)
        self.u_b = np.array(u_b)
        self.l_b2 = np.array(l_b2)
        self.u_b2 = np.array(u_b2)
        self.blur = blur
        self.bbox = None
        self.normalized_bbox = None

    def run(self):
        """
        Run the frame preprocessing and contour detection.

        Returns:
        - processed_frame (numpy.ndarray): Processed frame.
        - bbox (list): Bounding box coordinates(x, y, w, h).
        """
        # Preprocess the frame
        dilated = self.frame_preprocessor()
        # Find the biggest contour
        self.biggest_contour(dilated)
        self.normalize_bbox()
        return self.normalized_bbox

    def frame_preprocessor(self):
        """
        Preprocess a frame with a given color range and blur.

        Returns:
        - eroded (numpy.ndarray): Eroded mask.
        - dilated (numpy.ndarray): Dilated mask.
        """
        processed_frame = cv2.GaussianBlur(self.frame, self.blur, 0)
        processed_frame = self.color_isolation()

        eroded = cv2.erode(processed_frame, (3, 3), iterations=3)
        dilated = cv2.dilate(eroded, (3, 3), iterations=4)
        return dilated

    def color_isolation(self):
        """
        Perform color isolation on a frame.

        Returns:
        - processed_frame (numpy.ndarray): Frame after color isolation.
        """
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(self.frame, self.l_b, self.u_b)
        mask2 = cv2.inRange(self.frame, self.l_b2, self.u_b2)
        processed_frame = cv2.addWeighted(mask, 0.5, mask2, 0.5, 0)
        return processed_frame

    def biggest_contour(self, bool_image):
        """
        Find and adjust biggest contours based on a binary image.

        Parameters:
        - bool_image (numpy.ndarray): Binary image.

        Returns:
        - bbox (list): Bounding box coordinates(x, y, w, h).
        """
        contours, hierarchy = cv2.findContours(bool_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = tuple(sorted(contours, key=cv2.contourArea, reverse=True))
        if len(contours) >= 1:
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            self.bbox = [x, y, w, h]

        else:
            # if no contours(detected box) bbox will be none
            # can be used in checking if no box detected
            self.bbox = None

    # TODO
    def normalize_bbox(self):
        """
        """
        if self.bbox is not None:
            h, w, _ = self.frame.shape
            normalized_x, normalized_w, = self.bbox[0] / w, self.bbox[2] / w

            normalized_y, normalized_h = self.bbox[1] / h, self.bbox[3] / h

            self.normalized_bbox = [normalized_x, normalized_y, normalized_w, normalized_h]
        else:
            self.normalized_bbox = None