class DecisionMaker:
    """
    This class determines the directions that should be followed
    by the ROV in order to center it on top of the detected object.
    """

    def __init__(self, frame, bbox):
        """
        :param frame: input frame which has been read by the ROV camera
        :param bbox: bounding box of the detected object
        """
        self.frame = frame
        self.bbox = bbox
        self.frame_height, self.frame_width, _ = frame.shape
        self.frame_center_x = self.frame_width // 2
        self.frame_center_y = self.frame_height // 2
        self.bbox_center_x = (bbox[2] + bbox[0]) // 2
        self.bbox_center_y = (bbox[3] + bbox[1]) // 2

    def GetDirections(self):
        """
        Calculate and return movement vectors based on the position of the object.

        :return: Tuple containing X and Y movement vectors.
        """
        move_x = self.frame_center_x - self.bbox_center_x
        move_y = self.frame_center_y - self.bbox_center_y

        text = "Object in center"
        if abs(move_x) > 20 or abs(move_y) > 20:
            text = ""
            if move_x > 20:
                text = "Far left" if move_x > 50 else "A little bit to the left"
            elif move_x < -20:
                text = "Far right" if move_x < -50 else "A little bit to the right"
            if move_y > 20:
                text += ", Far up" if move_y > 50 else ", A little bit up"
            elif move_y < -20:
                text += ", Far down" if move_y < -50 else ", A little bit down"

        #Optionally: Use the `text` with accessories

        vector_x = round(move_x / self.frame_width, 2)
        vector_y = round(move_y / self.frame_height, 2)

        return vector_x, vector_y
