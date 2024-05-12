import cv2
from ultralytics import YOLO
import numpy as np
import cv2
import numpy as np
import torch
class ObjectDetection:
    """
    Class for object detection using YOLO model and calculating movement vectors and focal length.
    """

    def __init__(self, model_path):
        """
        Initialize the ObjectDetection class.

        Parameters:
        - model_path (str): Path to the YOLO model.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path)
        self.model.to(device) 

        self.focal_length = 1140
        self.size_of_black_square_in_real_world = 15.0

        self.arrow_max_length = 50
        self.arrow_thickness = 10
        self.arrow_color = (0, 0, 255)

        self.height = None
        self.width = None
        self.frame_center_x = None
        self.frame_center_y = None
        self.frame_center_point = None

    def DrawLine(self, start_point, end_point, color=(0, 0, 255), thickness=2):
        """
        Draw a line on the frame.

        Parameters:
        - start_point (tuple): Starting point of the line.
        - end_point (tuple): Ending point of the line.
        - color (tuple): Color of the line (BGR format).
        - thickness (int): Thickness of the line.
        """
        start_point = (int(start_point[0]), int(start_point[1]))
        end_point = (int(end_point[0]), int(end_point[1]))
        cv2.line(self.frame, start_point, end_point, color, thickness)

    def DrawArrow(self, direction, color=(0, 0, 255), thickness=2, max_length=50, padding=50):
        """
        Draw an arrow on the frame.

        Parameters:
        - direction (tuple): Direction vector of the arrow.
        - color (tuple): Color of the arrow (BGR format).
        - thickness (int): Thickness of the arrow.
        - max_length (int): Maximum length of the arrow.
        - padding (int): Padding from the frame edges.
        """
        arrow_start = (self.width - padding, padding)
        arrow_end = (arrow_start[0] - int(direction[0]), arrow_start[1] - int(direction[1]))
        arrow_length = min(max_length, np.linalg.norm(direction))
        scaled_direction = (direction / np.linalg.norm(direction)) * arrow_length
        cv2.arrowedLine(self.frame, arrow_start,(arrow_start[0] - int(scaled_direction[0]), arrow_start[1] - int(scaled_direction[1])), color,thickness, tipLength=0.5)

    def GetMovements(self, frameX, frameY, boxX, boxY, depth):
        """
        Calculate the movements and direction vectors.

        Parameters:
        - frameX (int): X-coordinate of the frame center.
        - frameY (int): Y-coordinate of the frame center.
        - boxX (int): X-coordinate of the box center.
        - boxY (int): Y-coordinate of the box center.
        - depth (float): Distance to the box.

        Returns:
        - vectors (list): List of movement vectors and depth.
        """
        move_x = frameX - boxX
        move_y = frameY - boxY
        text = ""
        if (move_x <= 20 and move_x >= -20) and (move_y <= 20 and move_y >= -20):
            text = "Object in center"
        else:
            if move_x > 0:
                text = "Far left" if move_x > 50 else "A little bit to the left"
            elif move_x < 0:
                text = "Far right" if move_x < -50 else "A little bit to the right"
            if move_y > 0:
                text += " Far up" if move_y > 50 else " A little bit up"
            elif move_y < 0:
                text += " Far down" if move_y < -50 else " A little bit down"

        cv2.putText(self.frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        vector_x = round(move_x / self.frame_center_x, 2)
        vector_y = round(move_y / self.frame_center_y, 2)
        vector_z = round(depth / 120, 2)

        vectors = [round(move_x,2), round(move_y,2), round(depth,2)]

        return vectors

    def ProcessFrame(self, frame):
        """
        Process a frame for object detection and calculate movements and direction vectors.

        Parameters:
        - frame (numpy.ndarray): Input frame.

        Returns:
        - frame (numpy.ndarray): Processed frame.
        """
        self.frame = frame
        if self.height is None or self.width is None:
            self.height, self.width, _ = frame.shape
            self.frame_center_x = self.width // 2
            self.frame_center_y = self.height // 2
            self.frame_center_point = (self.frame_center_x, self.frame_center_y)

        try:
            results = self.model.predict(frame, show=True, conf=0.95)
            num_objects = sum(len(result) for result in results)

            if num_objects > 1:
                cv2.putText(frame, "Too many objects to adjust to", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            elif num_objects == 0:
                cv2.putText(frame, "No Detections", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                for result_idx, result in enumerate(results):
                    for i, det in enumerate(result.boxes):
                        bbox = det.xyxy[0].tolist()
                        size_of_black_square_in_image = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

                        box_center_x = (bbox[2] + bbox[0]) // 2
                        box_center_y = (bbox[3] + bbox[1]) // 2

                        self.DrawLine(self.frame_center_point, (box_center_x, box_center_y))

                        distance = (self.size_of_black_square_in_real_world * self.focal_length) / size_of_black_square_in_image

                        direction_vector = np.array(self.frame_center_point) - np.array((box_center_x, box_center_y))

                        is_object_in_center = (abs(self.frame_center_x - box_center_x) <= 20) and (abs(self.frame_center_y - box_center_y) <= 20)
                        box_color = (255, 0, 0) if is_object_in_center else (0, 255, 0)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, 2)

                        label = f"Object {i + 1} : Class: Box, Distance: {distance:.2f} cm"
                        cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1, cv2.LINE_AA)

                        vectors = self.GetMovements(self.frame_center_x, self.frame_center_y, box_center_x, box_center_y, distance)
                        
                        print(vectors)

                self.DrawArrow(direction_vector, self.arrow_color, self.arrow_thickness, max_length=self.arrow_max_length)

        except Exception as e:
            print(f"An exception occurred: {e}")

        return frame

    def calculate_focal_length(self, object_distance_real):
        """
        Calculate the focal length using the lens formula.

        Parameters:
        - object_distance_real (float): Distance to the object in the real world.

        Returns:
        - None
        """
        cap = cv2.VideoCapture(0)
        capturing_frames = True

        while capturing_frames:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture a frame.")
                break

            cv2.imshow("Capture Frame", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                capturing_frames = False
            elif key == 27:
                break

        results = self.model.predict(frame, show=True, conf=0.99)

        for result_idx, result in enumerate(results):
            for i, det in enumerate(result.boxes):
                bbox = det.xyxy[0].tolist()
                self.size_of_black_square_in_image = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    
        self.focal_length = round((self.size_of_black_square_in_image * object_distance_real) / self.size_of_black_square_in_real_world)
        
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cap.release()
        
        print(f"The focal length is: {self.focal_length}")
    
if __name__ == "__main__":
    model_path = r"Models\best (1).pt"
    detection = ObjectDetection(model_path)

    # Call the calculate_focal_length function
    #detection.calculate_focal_length(43.0)

    choice = input("Choose input type (1 for live stream, 2 for video file): ")
    if choice == '1':
        cap = cv2.VideoCapture(0)  # Use live stream
    elif choice == '2':
        video_path = input("Enter the path to your video file: ")
        cap = cv2.VideoCapture(video_path)  # Use video file
    else:
        print("Invalid choice. Please enter either 1 or 2.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detection.ProcessFrame(frame)

        cv2.imshow("Object Detection", processed_frame)

        if cv2.waitKey(1) & 0XFF == ord('q'):
            break