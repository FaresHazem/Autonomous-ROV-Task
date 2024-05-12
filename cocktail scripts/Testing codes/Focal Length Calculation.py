import cv2
from ultralytics import YOLO
import numpy as np
class ObjectDetection:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.focal_length = 1140
        self.size_of_black_square_in_real_world = 15.0

    def calculate_focal_length(self, object_distance_real):
        # Open the camera
        cap = cv2.VideoCapture(0)
        
        # Boolean value to iterate on it
        capturing_frames = True

        while capturing_frames:
            # Capture a frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture a frame.")
                break

            # Display the captured frame for manual measurement of object width
            cv2.imshow("Capture Frame", frame)
            
            # Check for key press (press 'q' to exit and calculate focal length)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                capturing_frames = False
            elif key == 27:  # Press 'Esc' to exit without calculating focal length
                break

        # Perform object detection on the last captured frame
        results = self.model.predict(frame, show=True, conf=0.99)

        # Loop through the detected results
        for result_idx, result in enumerate(results):
            for i, det in enumerate(result.boxes):
                # Get bounding box
                bbox = det.xyxy[0].tolist()
                self.size_of_black_square_in_image = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    
        # Calculate the focal length using the lens formula
        self.focal_length = round((self.size_of_black_square_in_image * object_distance_real) / self.size_of_black_square_in_real_world)
        
        # Release the camera
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        cap.release()
        
        print(f"The focal length is: {self.focal_length}")
    
if __name__ == "__main__":
    model_path = r"C:\Users\fares\Desktop\Demo\best.pt"
    detection = ObjectDetection(model_path)

    # Call the calculate_focal_length function
    detection.calculate_focal_length(43.0)