import cv2
import numpy as np


class ParameterAdjustment:
    """
    Class for adjusting parameters of FramePreprocessor interactively.
    """

    def __init__(self, frame_preprocessor):
        """
        Initialize the ParameterAdjustment class.

        Parameters:
        - frame_preprocessor (FramePreprocessor): Instance of FramePreprocessor class.
        """
        self.frame_preprocessor = frame_preprocessor
        self._create_trackbars()

    def _empty(self, a):
        pass

    def _create_trackbars(self):
        """
        Create trackbars for parameter adjustments.
        """
        # Create a named window for trackbars
        cv2.namedWindow("Parameters")
        cv2.resizeWindow("Parameters", 450, 500)

        # Create trackbars for parameter adjustments
        cv2.createTrackbar("l_b1_hue_min", "Parameters", 0, 179, self._empty)
        cv2.createTrackbar("l_b1_saturation_min", "Parameters", 0, 255, self._empty)
        cv2.createTrackbar("l_b1_value_min", "Parameters", 0, 255, self._empty)
        cv2.createTrackbar("u_b1_hue_max", "Parameters", 179, 179, self._empty)
        cv2.createTrackbar("u_b1_saturation_max", "Parameters", 255, 255, self._empty)
        cv2.createTrackbar("u_b1_value_max", "Parameters", 255, 255, self._empty)
        cv2.createTrackbar("l_b2_hue_min", "Parameters", 0, 179, self._empty)
        cv2.createTrackbar("l_b2_saturation_min", "Parameters", 0, 255, self._empty)
        cv2.createTrackbar("l_b2_value_min", "Parameters", 0, 255, self._empty)
        cv2.createTrackbar("u_b2_hue_max", "Parameters", 179, 179, self._empty)
        cv2.createTrackbar("u_b2_saturation_max", "Parameters", 255, 255, self._empty)
        cv2.createTrackbar("u_b2_value_max", "Parameters", 255, 255, self._empty)
        cv2.createTrackbar("blur_size", "Parameters", 1, 30, self._empty)

    def run(self):
        """
        Callback function for trackbar changes.
        """
        # Get current trackbar values
        self.frame_preprocessor.l_b = np.array([cv2.getTrackbarPos("l_b1_hue_min", "Parameters"), cv2.getTrackbarPos("l_b1_saturation_min", "Parameters"), cv2.getTrackbarPos("l_b1_value_min", "Parameters")])
        self.frame_preprocessor.u_b = np.array([cv2.getTrackbarPos("u_b1_hue_max", "Parameters"), cv2.getTrackbarPos("u_b1_saturation_max", "Parameters"), cv2.getTrackbarPos("u_b1_hue_max", "Parameters")])
        self.frame_preprocessor.l_b2 = np.array([cv2.getTrackbarPos("l_b2_hue_min", "Parameters"), cv2.getTrackbarPos("l_b2_saturation_min", "Parameters"), cv2.getTrackbarPos("l_b2_value_min", "Parameters")])
        self.frame_preprocessor.u_b2 = np.array([cv2.getTrackbarPos("u_b2_hue_max", "Parameters"), cv2.getTrackbarPos("u_b2_saturation_max", "Parameters"), cv2.getTrackbarPos("u_b2_value_max", "Parameters")])
        self.frame_preprocessor.blur = np.array([cv2.getTrackbarPos("blur_size", "Parameters") * 2 + 1] * 2)
