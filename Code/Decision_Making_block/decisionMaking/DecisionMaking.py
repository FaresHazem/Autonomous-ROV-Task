import threading
import time

class DecisionMaker:
    """
    This class determines the directions that should be followed
    by the ROV in order to center it on top of the detected object.
    """

    def __init__(self, bbox=None):
        """
        Initialize the DecisionMaker object.

        :param bbox: Tuple containing x, y, width, and height of the bounding box
        """
        self.bbox = bbox
        self.bbox_center_x = None
        self.bbox_center_y = None
        self.vector_x, self.vector_y = 0.0, 0.0  # Initialize vector_y to 0.0


    def run(self):
        """
        Run the DecisionMaker and print the movement vector and optional arrow text.

        :return: Tuple containing X and Y movement vectors.
        """
        self.GetDirections()
        return self.vector_x, self.vector_y

    def calculate_bbox_center(self):
        """
        Calculate the center points of the bounding box.
        """
        x, y, w, h = self.bbox
        self.bbox_center_x = x + w + 1 / 2
        self.bbox_center_y = y + h + 1 / 2

    def GetDirections(self):
        """
        Calculate and return movement vectors based on the position of the object.

        :return: Tuple containing X and Y movement vectors.
        """
        if self.bbox is not None:
            self.calculate_bbox_center()
            self.vector_x = 0 - self.bbox_center_x
            self.vector_y = 0 - self.bbox_center_y
            self.vector_x = round(self.vector_x , 2)
            self.vector_y = round(self.vector_y , 2)