# Importing OpenCV for image processing and YOLOv8 for object detection.
import cv2

# Load the image
image = cv2.imread("img.jpg")

# Load YOLO model
from ultralytics import YOLO
model = YOLO("best.pt")

# Perform object detection
results = model.predict(image, show=True, conf=0.999)

# Assuming only one object is detected
bbox = results[0].boxes.xyxy.tolist()[0]

# Calculate the size in pixels
size_of_black_square_in_image = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

# Print the calculated size
print(size_of_black_square_in_image)