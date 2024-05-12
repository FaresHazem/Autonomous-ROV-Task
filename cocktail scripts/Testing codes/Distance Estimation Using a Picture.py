# Importing the YOLOv8 library and OpenCV for object detection and distance estimation.
from ultralytics import YOLO
import cv2

# Load the image
image = cv2.imread(r"D:\College\External Courses\CrocoMarine ROV Competitions\Final Project\Code\Images\img.jpg")

# Load YOLO model
model = YOLO(r"D:\College\External Courses\CrocoMarine ROV Competitions\Final Project\Code\Models\best.pt")

# Perform object detection
results = model.predict(image, show=True, conf=0.999)

# Assuming only one object is detected
bbox = results[0].boxes.xyxy.tolist()[0]

# Calculate the size in pixels
size_of_black_square_in_image = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

# Focal length and size of the black square in real-world units
focal_length = 1000  
size_of_black_square_in_real_world = 15.0  

# Calculate distance using the pinhole camera model
distance = (size_of_black_square_in_real_world * focal_length) / size_of_black_square_in_image

# Print the estimated distance
print(f"Estimated distance to the black square: {distance} cm")