#blurring and smoothin
import cv2
import numpy as np
import cv2
import numpy as np
import cv2
import numpy as np

class ColorIsolation:
    """
    A class that performs color isolation on an image.

    Attributes:
        img1 (numpy.ndarray): The input image.

    Methods:
        isolate_color(): Applies color isolation on the image and returns the result.
    """

    def __init__(self, img1):
        """
        Initializes a new instance of the ColorIsolation class.

        Args:
            img1 (numpy.ndarray): The input image.
        """
        self.img1 = img1

    def isolate_color(self):
        """
        Applies color isolation on the image and returns the result.

        Returns:
            numpy.ndarray: The color isolated image.
        """
        # Convert image to grayscale and HSV color space
        gray = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(self.img1, cv2.COLOR_BGR2HSV)

        # Define lower and upper red color ranges
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])

        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color ranges
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(self.img1, self.img1, mask=mask)

        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        res2 = cv2.bitwise_and(self.img1, self.img1, mask=mask2)

        # Combine the color isolated images
        img3 = res + res2
        img4 = cv2.add(res, res2)
        img5 = cv2.addWeighted(res, 0.5, res2, 0.5, 0)

        # Apply smoothing filter to the color isolated images
        kernel = np.ones((15, 15), np.float32) / 225
        smoothed = cv2.filter2D(res, -1, kernel)
        smoothed2 = cv2.filter2D(img3, -1, kernel)

        return img5

# Example usage:
img1 = cv2.imread('saved_3.jpg', 1)
color_isolation = ColorIsolation(img1)
smoothed2 = color_isolation.isolate_color()



cv2.imshow('Original',img1)
#cv2.imshow('Original1',img1[:,:,1])
#cv2.imshow('Original2',img1[:,:,2])

# cv2.imshow('Averaging',smoothed)
# cv2.imshow('mask',mask)
# cv2.imshow('res',res)
# cv2.imshow('mask2',mask2)
# cv2.imshow('res2',res2)
# cv2.imshow('res3',img3)
# cv2.imshow('res4',img4)
# cv2.imshow('res5',img5)
cv2.imshow('smooth2',smoothed2)




cv2.waitKey(0)
cv2.destroyAllWindows()