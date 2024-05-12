import threading
import time


class DistanceEstimator:

    """
    This class estimates the distance between the detected box and the camera lens
    """

    def __init__(self, size_of_box=15.0, focal_length=1140, bbox=None):
        """
        :param size_of_box: size of box in read world
        :param focal_length: focal length of the camera
        :param bbox: bounding box of the detected object
        """
        self.size_of_box = size_of_box
        self.focal_length = focal_length
        self.bbox = bbox
        # Thread
        self.no_bbox_thread = threading.Thread(target=self.check_bbox_detection)
        self.no_bbox_thread.daemon = True
        self.no_bbox_thread.start()
        self.bbox_detected = True  # flag to indicate whether the bounding box is detected
        self.update_vector_z = False  # flag to indicate whether to update vector_y

        self.distance = 0.0


    def run(self):
        """
        Run the DecisionMaker and print the movement vector and optional arrow text.

        :return: Tuple containing X and Y movement vectors.
        """
        self.Calculate()
        return self.distance

    def Calculate(self):
        """
        Using the bbox we estimate the distance

        :return: Normalized distance to the detected box between -2 and 2
        """

        if self.bbox is not None:
            size_of_box_in_image = (max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])) + 1
            if(size_of_box_in_image == 0):
                print(self.bbox)
                size_of_box_in_image = 1
            self.distance = (self.size_of_box * self.focal_length) / size_of_box_in_image
            self.distance = round(self.distance / 100, 2)
            self.distance = max(min(self.distance, 2), -2)
        else:
            self.no_box_detected()

    def no_box_detected(self):
        """
        Function to execute when no box is detected.
        Modifies vector_x and vector_y accordingly.
        """
        if (self.update_vector_z) and ((self.distance < 1.0) and (self.distance > -1.0)):
            self.distance += 0.05
            self.update_vector_z = False

    def check_bbox_detection(self):
        """
        Thread to check if the bounding box is detected,
        if not detected then we increase vector_y until the bounding box is detected again.
        """
        while True:
            if self.bbox is None:
                self.bbox_detected = False
                self.update_vector_z = True
            else:
                self.bbox_detected = True
            time.sleep(3)