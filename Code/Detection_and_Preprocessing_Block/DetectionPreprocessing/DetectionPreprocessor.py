from abc import ABC, abstractmethod
class DetectionPreprocessor(ABC):
    """
    Abstract base class for a Detection Preprocessor.

    This class is designed to be subclassed by concrete detection
    preprocessors. Subclasses should implement the `run()` method,
    which performs the detection preprocessing and returns the bounding
    box of the detected object.

    Attributes:
        frame (numpy.ndarray): The input frame.
        bbox (list or None): The bounding box of the detected object.
            It should be `None` if no box is detected.

    Args:
        frame (numpy.ndarray): The input frame.

    """
    def __init__(self, frame=None):
        """
        Initialize the DetectionPreprocessor object with the input frame.

        Args:
            frame (numpy.ndarray): The input frame.
        """
        self.frame = frame
        self.bbox = None

    @abstractmethod
    def run(self):
        """
        This function runs the DetectionPreprocessor module and returns the
        bounding box of the detected object.

        The bounding box should be `None` if no box is detected (can be used in
        conditioning).

        Returns:
            list or None: Bounding box of the detected object (x, y, w, h), or
            `None` if no box is detected.

        """
        return self.bbox
