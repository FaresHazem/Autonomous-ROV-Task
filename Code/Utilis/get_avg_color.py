import cv2
import numpy as np 

def get_avg_color(frame, bbox):
    """
    Calculate the average color of a region in an RGB frame.

    Parameters:
    - frame: numpy array of shape (height, width, channels)
    - bbox: list of [x, y, w, h] coordinates of the region

    Returns:
    - average_color: list of [H, S, V] values of the region
    """

    # Calculate the coordinates and dimensions of the region
    x = int((bbox[0] + 1) * frame.shape[1] / 2)
    y = int((bbox[1] + 1) * frame.shape[0] / 2)
    w = int((bbox[2] + 2) * frame.shape[1] / 2)
    h = int((bbox[3] + 2) * frame.shape[0] / 2)

    # Crop the region from the frame
    croped_frame = frame[y:y+h, x:x+w]

    # Convert the cropped frame to HSV color space
    hsv = cv2.cvtColor(croped_frame, cv2.COLOR_RGB2HSV)

    # Calculate the average color of the region
    average_color = [np.mean(hsv[:, :, i]) for i in range(3)]
    lower = np.array([average_color[0] - 10, average_color[1] - 10, average_color[2] - 10])
    upper = np.array([average_color[0] + 10, average_color[1] + 10, average_color[2] + 10])
    return lower, upper

# Test the function
if __name__ == "__main__":
    # Load the video
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    # Select the bounding box using mouse click
    bbox = cv2.selectROI("Select ROI", frame)

    # Initialize the tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker
        success, bbox = tracker.update(frame)

        if success:
            # Draw the bounding box on the frame
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Tracking", frame)

        # Exit if ESC key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()