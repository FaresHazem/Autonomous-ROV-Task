class DistanceEstimator:

    """
    This class estimates the distance between the detected box and the camera lens
    """

    def __init__(self, size_of_box, focal_length, bbox):
        """
        :param size_of_box: size of box in read world
        :param focal_length: focal length of the camera
        :param bbox: bounding box of the detected object
        """
        self.size_of_box = size_of_box
        self.focal_length = focal_length
        self.bbox = bbox


    def Calculate(self):

        """
        :return: Estimated distance to the detected box
        """

        if(self.bbox != None):
            size_of_box_in_image = max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])
            distance = (self.size_of_box * self.focal_length) / size_of_box_in_image
            distance = round(distance / 120, 2)

            return distance
        else:
            return None

