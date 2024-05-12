# Importing the YOLOv8 library and OpenCV for object prediction.
from ultralytics import YOLO
import cv2

# Initializing the YOLOv8 model with the specified weights file.
model = YOLO("best.pt")

# Performing object prediction on the specified source with a confidence threshold of 0.999.
model.predict(source="0", show=True, conf=0.999)