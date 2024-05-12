import cv2
import numpy as np
import sys
sys.path.insert(0, '..')
from ..DetectionPreprocessing.DetectionPreprocessor import *

class YoloDetector(DetectionPreprocessor):
    """
    """

    def __init__(self, frame=None):
        """
        Initialize the FramePreprocessor class.

        Parameters:
        - frame (numpy.ndarray): Input frame.
        """
        self.frame = frame
        self.bbox = None

    def run(self):
        """
        Run the frame preprocessing and contour detection.

        Returns:
        - processed_frame (numpy.ndarray): Processed frame.
        - bbox (list): Bounding box coordinates(x, y, w, h).
        """
        return self.bbox
