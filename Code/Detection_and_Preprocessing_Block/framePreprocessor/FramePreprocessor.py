import cv2
import cvzone
import numpy as np
from ..DetectionPreprocessing import DetectionPreprocessor

class FramePreprocessor(DetectionPreprocessor):
    """
    Class for preprocessing a frame with a given color range and blur.
    """

    def __init__(self, frame=None, 
    l_b  = [0, 50, 50], 
    u_b  = [10, 255, 255], 
    l_b2 = [155,23,169], 
    u_b2 = [255,255,255], 
    blur = (11, 11)):
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
        self.l_b  = np.array(l_b)
        self.u_b  = np.array(u_b)
        self.l_b2 = np.array(l_b2)
        self.u_b2 = np.array(u_b2)
        self.blur = blur
        self.bbox = None
        #self.masker = Masker(self.frame, [self.l_b, self.u_b], [self.l_b2, self.u_b2], self.blur)

    def run(self):
        """
        Run the frame preprocessing and contour detection.

        Returns:
        - processed_frame (numpy.ndarray): Processed frame.
        - bbox (list): Bounding box coordinates(x, y, w, h).
        """
        processed_mask = self.frame_preprocessor()
        self.biggest_contour(processed_mask)
        self.normalize_bbox()
        return self.normalized_bbox

    def frame_preprocessor(self):
        """
        Preprocess a frame with a given color range and blur.

        Returns:
        - Processed Frame: Mask
        """
        processed_frame = cv2.GaussianBlur(self.frame, self.blur, 0)
        processed_frame = self.color_isolation(processed_frame)
        return processed_frame

    def color_isolation(self, frame):
        """
        Perform color isolation on a frame.

        Returns:
        - processed_frame (numpy.ndarray): Frame after color isolation.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(frame,  self.l_b, self.u_b)
        mask2 = cv2.inRange(frame, self.l_b2, self.u_b2)
        processed_frame = cv2.addWeighted(mask1, 0.5, mask2, 0.5, 0)

        processed_frame = cv2.erode(processed_frame, (5, 5), iterations=3)
        processed_frame = cv2.dilate(processed_frame, (5, 5), iterations=3)
        cv2.imshow("Frame", processed_frame)
        return processed_frame

    def biggest_contour(self, mask):
        """
        Find and adjust biggest contours based on a binary image.

        Parameters:
        - bool_image (numpy.ndarray): Binary image.

        Returns:
        - bbox (list): Bounding box coordinates(x, y, w, h).
        """
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = tuple(sorted(contours, key=cv2.contourArea, reverse=True))
        if len(contours) >= 1:
            cnt = contours[0]
            x, y, w, h = cv2.boundingRect(cnt)
            self.bbox = [x, y, w, h]
            area = (w * h)
            if(area > 300):
                self.bbox = [x, y, w, h]
            else:
                self.bbox = None
        else:
           #no contours detected
            self.bbox = None

    def normalize_bbox(self):
        """
        Normalize the bounding box coordinates to the range [-1, 1].
        If the bounding box is not None, normalize its coordinates based on the frame shape.
        """
        if self.bbox is not None:
            h, w, _ = self.frame.shape
            normalized_x = (self.bbox[0] * 2 / w) - 1
            normalized_w = (self.bbox[2] * 2 / w) - 1
            normalized_y = (self.bbox[1] * 2 / h) - 1
            normalized_h = (self.bbox[3] * 2 / h) - 1

            self.normalized_bbox = [normalized_x, normalized_y, normalized_w, normalized_h]
        else:
            self.normalized_bbox = None