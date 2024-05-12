import cv2
from ultralytics import YOLO
import numpy as np
import os
from Detection_and_Preprocessing_Block.framePreprocessor.FramePreprocessor import * 

class ObjectDetection:
    """
    Class for object detection using YOLO model and calculating movement vectors and focal length.
    """

    def __init__(self):
        """
        Initialize the ObjectDetection class.
        """

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

        self.frame = None
    #=========================== PUT IN A SEPARATE CLASS IN FOLDER UTILIS======# 
    def DrawLine(self, start_point, end_point, color=(0, 0, 255), thickness=2):
        """
        Draw a line on the frame.

        Parameters:
        - start_point (tuple): Starting point of the line.
        - end_point (tuple): Ending point of the line.
        - color (tuple): Color of the line (BGR format).
        - thickness (int): Thickness of the line.
        """
        start_point = (int(640), int(480))
        end_point = (int(end_point[0]), int(end_point[1]))
        cv2.line(self.frame, start_point, end_point, color, thickness)

    #=========================== PUT IN A SEPARATE CLASS IN FOLDER UTILIS======# 
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
        cv2.arrowedLine(self.frame, arrow_start, (arrow_start[0] - int(scaled_direction[0]), arrow_start[1] - int(scaled_direction[1])), color, thickness, tipLength=0.5)
    #=========================== PUT IN A SEPARATE CLASS IN FOLDER UTILIS======# 
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

        vectors = [round(move_x, 2), round(move_y, 2), round(depth, 2)]

        return vectors

    def Accessories(self, num_objects=None, bbox=None, distance=None, direction_vector=None):
        """
        Add accessories such as text and shapes to the frame.

        Parameters:
        - frame (numpy.ndarray): Input frame.
        - num_objects (int): Number of objects detected.
        - bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max].
        - distance (float): Distance to the object.
        - direction_vector (tuple): Direction vector for arrow drawing.

        Returns:
        - None
        """
        if num_objects is not None:
            if num_objects > 1:
                cv2.putText(self.frame, "Too many objects to adjust to", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            elif num_objects == 0:
                cv2.putText(self.frame, "No Detections", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        if bbox is not None:
            # Calculate box center
            box_center_x = (bbox[2] + bbox[0]) // 2
            box_center_y = (bbox[3] + bbox[1]) // 2
            print(box_center_x)
            # Draw line from frame center to box center
            #self.DrawLine(self.frame_center_point, [box_center_x, box_center_y])

            # Calculate distance
            size_of_black_square_in_image = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            distance = (self.size_of_black_square_in_real_world * self.focal_length) / size_of_black_square_in_image

            # Calculate direction vector
            direction_vector = np.array(self.frame_center_point) - np.array((box_center_x, box_center_y))

            # Check if object is in the center
            is_object_in_center = (abs(self.frame_center_x - box_center_x) <= 20) and (abs(self.frame_center_y - box_center_y) <= 20)
            box_color = (255, 0, 0) if is_object_in_center else (0, 255, 0)
            
            # Draw rectangle around the object
            cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, 2)

            # Calculate and add movement text
            vectors = self.GetMovements(self.frame_center_x, self.frame_center_y, box_center_x, box_center_y, distance)
            #move_text = f"Move_X: {vectors[0]}, Move_Y: {vectors[1]}, Depth: {vectors[2]}"
            #cv2.putText(frame, move_text, (int(bbox[0]), int(bbox[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        if direction_vector is not None:
            # Draw arrow based on direction vector
            self.DrawArrow(direction_vector, self.arrow_color, self.arrow_thickness, max_length=self.arrow_max_length)

        if distance is not None:
            # Add label with distance
            label = f"Box : Class: Box, Distance: {distance:.2f} cm"
            cv2.putText(self.frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


    def YoloModel(self):
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

            for result_idx, result in enumerate(results):
                for i, det in enumerate(result.boxes):
                    bbox = det.xyxy[0].tolist()

                    self.Accessories(frame, num_objects=num_objects, bbox=bbox)

        except Exception as e:
            print(f"An exception occurred: {e}")

        return frame

    def ImageProcessing(self):
        """
        Process a frame for object detection and calculate movements and direction vectors.

        Parameters:
        - frame (numpy.ndarray): Input frame.

        Returns:
        - frame (numpy.ndarray): Processed frame.
        """
        processor = FramePreprocessor(frame= self.frame)
        bbox = processor.run()
        if bbox is not None:
            x,y,w,h = bbox
            bbox = [int(x * 640), int(y*480), int((x + w)*640), int((y + h)*480)]
            print(bbox)
            cv2.rectangle(self.frame, (int(x * 640),int( y * 480)), (int((x + w) * 640), int((y + h )* 480)), 255, 2)
            #self.Accessories(bbox= bbox)

        return self.frame


    def Menu(self):
        """
        Menu to choose input type and processing method.
        """
        print("Choose input type:")
        print("1. Live stream")
        print("2. Video file")
        #input_type = input("Enter your choice: ")
        input_type = "2"

        if input_type == '1':
            # Live stream
            cap = cv2.VideoCapture(0)
        elif input_type == '2':
            # Video file
            #video_path = input("Enter the path of the video file: ").strip('"')  # Remove double quotes
            video_path = "..\Data\Videos\Sample 5.mp4"
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print("Error: Unable to open video file.")
                    return
            except Exception as e:
                print(f"Error: {e}")
                return
        else:
            print("Invalid input type.")
            return

        print("Choose processing method:")
        print("1. YOLO Model")
        print("2. Image Processing")
        #processing_method = input("Enter your choice: ")
        processing_method = "2"
        while True:
            ret, self.frame = cap.read()
            if not ret:
                print("No frame captured.")
                break
            
            if processing_method == '1':
                # Process frame using YOLO model
                #processed_frame = self.YoloModel(frame)
                pass
            elif processing_method == '2':
                # Process frame using image processing
                processed_frame = self.ImageProcessing()
            
            cv2.imshow("Object Detection", processed_frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:  # 27 is the ASCII code for 'Esc'
                print("Exiting...")
                break
            elif key == ord('s'):  # Capture and save screenshot
                cv2.imwrite(f"Frames/frame_{len(os.listdir('Frames'))}.png", processed_frame)
                # if mask_frame is not None:
                #     cv2.imwrite(f"Masks/mask_{len(os.listdir('Masks'))}.png", mask_frame)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detection = ObjectDetection()
    detection.Menu()
