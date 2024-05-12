import cv2
from ultralytics import YOLO
import numpy as np
import pyttsx3

class ObjectDetection:
    def __init__(self, model_path, video_source=0):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_source)
        self.engine = pyttsx3.init()

        # Constants for camera calibration
        self.focal_length = 1140
        self.size_of_black_square_in_real_world = 15.0

        # Arrow parameters
        self.arrow_max_length = 50
        self.arrow_thickness = 10
        self.arrow_color = (0, 0, 255)

        # Initialize frame properties
        ret, frame = self.cap.read()
        self.height, self.width, _ = frame.shape
        self.frame_center_x = self.width // 2
        self.frame_center_y = self.height // 2
        self.frame_center_point = (self.frame_center_x, self.frame_center_y)

    def draw_line(self, start_point, end_point, color=(0, 0, 255), thickness=2):
        start_point = (int(start_point[0]), int(start_point[1]))
        end_point = (int(end_point[0]), int(end_point[1]))
        cv2.line(self.frame, start_point, end_point, color, thickness)

    def draw_arrow(self, direction, color=(0, 0, 255), thickness=2, max_length=50, padding=50):
        arrow_start = (self.width - padding, padding)
        arrow_end = (arrow_start[0] - int(direction[0]), arrow_start[1] - int(direction[1]))

        arrow_length = min(max_length, np.linalg.norm(direction))
        scaled_direction = (direction / np.linalg.norm(direction)) * arrow_length

        cv2.arrowedLine(self.frame, arrow_start, (arrow_start[0] - int(scaled_direction[0]), arrow_start[1] - int(scaled_direction[1])), color, thickness, tipLength=0.5)

    def movements(self, frameX, frameY, boxX, boxY):
        move_x = frameX - boxX
        move_y = frameY - boxY

        text = ""

        if (move_x <= 20 and move_x >= -20) and (move_y <= 20 and move_y >= -20):
            text = "Object in center"
            self.engine.say(text)
            self.engine.runAndWait()
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

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            try:
                results = self.model.predict(self.frame, show=True, conf=0.999)
                num_objects = sum(len(result) for result in results)

                if num_objects > 1:
                    cv2.putText(self.frame, "Too many objects to adjust to", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                elif num_objects == 0:
                    cv2.putText(self.frame, "No Detections", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    for result_idx, result in enumerate(results):
                        for i, det in enumerate(result.boxes):
                            bbox = det.xyxy[0].tolist()
                            size_of_black_square_in_image = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                            box_center_x = (bbox[2] + bbox[0]) // 2
                            box_center_y = (bbox[3] + bbox[1]) // 2
                            self.draw_line(self.frame_center_point, (box_center_x, box_center_y))

                            distance = (self.size_of_black_square_in_real_world * self.focal_length) / size_of_black_square_in_image
                            direction_vector = np.array(self.frame_center_point) - np.array((box_center_x, box_center_y))

                            is_object_in_center = (abs(self.frame_center_x - box_center_x) <= 20) and (abs(self.frame_center_y - box_center_y) <= 20)
                            box_color = (255, 0, 0) if is_object_in_center else (0, 255, 0)
                            cv2.rectangle(self.frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, 2)

                            label = f"Object {i + 1} : Class: Box, Distance: {distance:.2f} cm"
                            cv2.putText(self.frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            self.movements(self.frame_center_x, self.frame_center_y, box_center_x, box_center_y)

                    self.draw_arrow(direction_vector, self.arrow_color, self.arrow_thickness, max_length=self.arrow_max_length)

            except Exception as e:
                print(f"An exception occurred: {e}")

            cv2.imshow("Object Detection", self.frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detection = ObjectDetection(model_path=r"D:\College\External Courses\CrocoMarine ROV Competitions\Distance_Estimation_Using_YOLOv8\Code\Models\best.pt")
    detection.run()