import cv2
import numpy as np

class Masker:
    def __init__(self, frame=None, color_bound1=None, color_bound2=None, blur=None):
        self.frame = frame
        self.color_bound1 = color_bound1
        self.color_bound2 = color_bound2
        self.blur = blur

        self.merged_mask = None

    def run(self):
        if((self.frame is not None) and (self.color_bound1 is not None) and (self.color_bound2 is not None)):
            if self.blur != None:
                self.frame = cv2.GaussianBlur(self.frame, (self.blur), 0)

            hsv_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            lower_bound1 = np.array(self.color_bound1[0], dtype=np.uint8)
            upper_bound1 = np.array(self.color_bound1[1], dtype=np.uint8)
            lower_bound2 = np.array(self.color_bound2[0], dtype=np.uint8)
            upper_bound2 = np.array(self.color_bound2[1], dtype=np.uint8)

            mask1 = cv2.inRange(hsv_frame, lower_bound1, upper_bound1)
            mask1 = cv2.erode(mask1, None, iterations=2)
            mask1 = cv2.dilate(mask1, None, iterations=2)

            mask2 = cv2.inRange(hsv_frame, lower_bound2, upper_bound2)
            mask2 = cv2.erode(mask2, None, iterations=2)
            mask2 = cv2.dilate(mask2, None, iterations=2)

            merged_mask = cv2.bitwise_or(mask1, mask2)

        return merged_mask, [mask1, mask2]