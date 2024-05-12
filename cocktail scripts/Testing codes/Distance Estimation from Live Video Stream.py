# Import necessary libraries
import cv2
from ultralytics import YOLO

# Constants for camera calibration
focal_length = 1140  # Focal length of the camera
size_of_black_square_in_real_world = 15.0  # Size of the black square in the real world (in centimeters)

# Load YOLO model
model = YOLO(r"D:\College\External Courses\CrocoMarine ROV Competitions\Distance_Estimation_Using_YOLOv8\Code\Models\best.pt")

# Open a video capture object (using the default camera, change the argument if using a different camera)
cap = cv2.VideoCapture(0)

# Main loop for continuous video capture and object detection
while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()

    # Perform object detection using YOLO
    try:
        # Get detection results and display them
        results = model.predict(frame, show=True, conf=0.999)

        # Iterate over all detected objects
        for result_idx, result in enumerate(results):
            # Iterate over the bounding boxes using result.boxes
            for i, det in enumerate(result.boxes):
                # Get bounding box coordinates
                bbox = det.xyxy[0].tolist()

                # Calculate the size of the black square in the image
                size_of_black_square_in_image = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

                # Calculate the estimated distance to the black square using camera calibration
                distance = (size_of_black_square_in_real_world * focal_length) / size_of_black_square_in_image

                # Display the bounding box on the video frame
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

                # Display the label with class and estimated distance on the video frame
                label = f"Object {i + 1} : Class: Box, Distance: {distance:.2f} cm"

                # Attach the label to the top-left corner of the bounding box
                cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    except IndexError:
        # Handle the case when no object is detected
        print("No detections found.")

    # Show the video frame with the bounding box and label
    cv2.imshow("Object Detection", frame)

    # Check for the 'Esc' key press to exit the loop
    key = cv2.waitKey(1)
    if key == 27:  # ASCII code for 'Esc' key
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
