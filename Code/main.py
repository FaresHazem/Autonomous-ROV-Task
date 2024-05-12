import cv2
import cvzone
from Detection_and_Preprocessing_Block import FramePreprocessor
from Decision_Making_block import DecisionMaker, DistanceEstimator
from Utilis import CameraHandler, VideoSaver, get_avg_color

# Hyperparameters
RUN = True  # Apply the code to the frames
DRAW = True  # Show basic UI to the user
CAPTURE = False # Capture the frames or not
ADJUST_PARAMETERS = False  # To manually adjust the color bounds

def Run(frame, processor=None, dmaker=None, estimator=None, adjuster=None):
    """
    Process a single frame by applying the predefined processing pipeline,
    which includes detection, decision making, and distance estimation.

    Args:
        frame (numpy.ndarray): The input frame to be processed.
        processor (FramePreprocessor, optional): The frame preprocessor object. Defaults to None.
        dmaker (DecisionMaker, optional): The decision maker object. Defaults to None.
        estimator (DistanceEstimator, optional): The distance estimator object. Defaults to None.
        adjuster (ParameterAdjustment, optional): The parameter adjuster object. Defaults to None.

    Returns:
        numpy.ndarray or None: The processed frame if successful, otherwise None.
    """
    # Extract frame dimensions
    frame_height, frame_width,_ = frame.shape

    # If RUN flag is set to True, process the frame
    if RUN:
        # Update the frame in the processor
        processor.frame = frame

        # Run the frame preprocessor and get the bounding box
        bbox = processor.run()

        # If a bounding box is obtained, draw it on the frame
        if bbox is not None:
            x, y, w, h = bbox
            if DRAW:
                cv2.rectangle(frame, (int((x + 1) * frame_width / 2), int((y + 1) * frame_height / 2)),
                            (int((x + w + 2) * frame_width / 2), int((y + h + 2) * frame_height / 2)), 255, 2)
        
            print(get_avg_color(frame, bbox))
        # Update the bounding box in the decision maker
        dmaker.bbox = bbox

        # Update the bounding box in the distance estimator
        estimator.bbox = processor.bbox

        # Run the decision maker and get the movement vector
        movement_vector_x, movement_vector_y = dmaker.run()

        # Run the distance estimator and get the estimated distance
        movement_vector_z = estimator.run()

        # If ADJUST_PARAMETERS flag is set to True, run the parameter adjuster
        if(ADJUST_PARAMETERS):
            adjuster.run()

        # If DRAW flag is set to True, print and draw the movement vector
        if DRAW:
            print(f"Vector-X: {movement_vector_x} | Vector-Y: {movement_vector_y} | Vector-Z: {movement_vector_z}")
            cv2.arrowedLine(frame, (320, 240), (int(320 - movement_vector_x * 50), int(240 - movement_vector_z * 50)),(0, 255, 0), 4)

            # Show the processed frame
            cv2.imshow("Processed Frame", frame)

if __name__ == "__main__":
    processor    = FramePreprocessor()
    dmaker       = DecisionMaker()
    estimator    = DistanceEstimator(size_of_box=15.0, focal_length=1140, bbox=None)
    if(ADJUST_PARAMETERS):
        adjuster = ParameterAdjustment(processor)
    else:
        adjuster = None
    camera = CameraHandler("Digital", 0)

    if CAPTURE:
        video_name = input("Enter the name for the saved video file: ")
        saver = VideoSaver(video_name + ".avi", frames=30)
    else:
        saver = None

    while True:
        # Process frames from the camera
        # Read the next frame from the camera
        timestamp, frame = camera.read()
        # If no frame was read, continue to the next iteration
        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        Run(frame.copy(), processor=processor, dmaker=dmaker, estimator=estimator)
        # Show the frames
        cv2.imshow("Frame", frame)

        # If a saver object is provided, save the frame
        if CAPTURE:
            saver.save_frame(frame)
        # If 'q' is pressed, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy the windows
    camera.release()
    cv2.destroyAllWindows()
    if CAPTURE:
        saver.release()
