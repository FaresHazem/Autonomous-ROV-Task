import cv2
from ultralytics import YOLO
import numpy as np
import os
from FrameProcessor import *
from DistanceEstimator import *
from DecisionMaker import *

class Autonomous:

    def __init__(self, focal_length, size_of_box):

        self.frame = None
        self.focal_length = focal_length
        self.size_of_box_in_real_world = size_of_box

        self.frame_processor = FramePreprocessor
        self.distance_estimator = DistanceEstimator
        self.decision_maker = DecisionMaker

    def YoloModel(self):
        """
        Process a frame for object detection and calculate movements and direction vectors.

        Parameters:
        - frame (numpy.ndarray): Input frame.

        Returns:
        - frame (numpy.ndarray): Processed frame.
        """

        if(self.frame != None):
            try:
                results = self.model.predict(self.frame, show=True, conf=0.95)
                num_objects = sum(len(result) for result in results)

                for result_idx, result in enumerate(results):
                    for i, det in enumerate(result.boxes):
                        bbox = det.xyxy[0].tolist()

                        #self.Accessories(frame, num_objects=num_objects, bbox=bbox)

            except Exception as e:
                print(f"An exception occurred: {e}")

        return self.frame, bbox, num_objects

    def ImageProcessing(self):
        """
        Process a frame for object detection and calculate movements and direction vectors.

        Parameters:
        - frame (numpy.ndarray): Input frame.

        Returns:
        - frame (numpy.ndarray): Processed frame.
        """
        bbox = self.frame_processor(frame=self.frame).run()
        if bbox is not None:
            x, y, w, h = bbox
            bbox = [int(x * 640), int(y * 480), int((x + w) * 640), int((y + h) * 480)]
            print(bbox)
            cv2.rectangle(self.frame, (int(x * 640), int(y * 480)), (int((x + w) * 640), int((y + h) * 480)), 255, 2)

        return self.frame, bbox

    def Menu(self):
        """
        Menu to choose input type and processing method.
        """
        print("Choose input type:")
        print("1. Live stream")
        print("2. Video file")
        # input_type = input("Enter your choice: ")
        input_type = "2"

        if input_type == '1':
            # Live stream
            cap = cv2.VideoCapture(0)
        elif input_type == '2':
            # Video file
            # video_path = input("Enter the path of the video file: ").strip('"')  # Remove double quotes
            video_path = "Data/Underwater.avi"
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
        # processing_method = input("Enter your choice: ")
        processing_method = "2"
        while True:
            ret, self.frame = cap.read()
            if not ret:
                print("No frame captured.")
                break

            if processing_method == '1':
                # Process frame using YOLO model
                # processed_frame = self.YoloModel(frame)
                pass
            elif processing_method == '2':
                # Process frame using image processing
                processed_frame, bbox = self.ImageProcessing()
                if(bbox != None):
                    distance = self.distance_estimator(size_of_box=self.size_of_box_in_real_world, focal_length=self.focal_length, bbox=bbox).Calculate()
                    vectors = self.decision_maker(frame=self.frame, bbox=bbox).GetDirections()
                    print(distance, vectors)

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
    detection = Autonomous(focal_length=1140, size_of_box=15.0)
    detection.Menu()