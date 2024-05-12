import cv2
import numpy as np
from Detection_and_Preprocessing_Block import FramePreprocessor
from Decision_Making_block import DecisionMaker, DistanceEstimator
import time

def calculate_execution_time(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of {function.__name__}: {execution_time} seconds")
    return result

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def test_frame_preprocessor():
    # Create a video capture object
    cap = cv2.VideoCapture("../Data/Videos/Sample 5.mp4")

    #blank_image_for_drawing = np.zeros((360,480), np.uint8)

    # Create an instance of FramePreprocessor
    preprocessor = FramePreprocessor()
    dmaker = DecisionMaker()
    #new distance estimator
    estimator = DistanceEstimator(size_of_box=15.0, focal_length=1140, bbox=None)
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        if not ret:
            pass

        frame_ = cv2.resize(frame,(640,480))
        frame = hisEqulColor(frame_.copy())
        preprocessor.frame = frame
        # Preprocess the frame using FramePreprocessor
        bbox = calculate_execution_time(preprocessor.run)  
        print(bbox)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (int((x + 1) * 640 / 2),int( (y + 1) * 480/2)), (int((x + w + 2) * 640/2), int((y + h + 2)* 480 / 2)), 255, 2)
        
        dmaker.bbox = bbox
        estimator.bbox = bbox
        movement_vector_x, movement_vector_y = dmaker.run()
        estimated_distance = estimator.Calculate()

        print(f"Vector-X: {movement_vector_x} | Vector-Y: {movement_vector_y}")
        print(f"Estimated Distance: {estimated_distance}")

        cv2.arrowedLine(frame, (320, 240), (int(320 - movement_vector_x * 50), int(240 - movement_vector_y * 50)), (0, 255, 0), 4)
        # Display the preprocessed frames
        cv2.imshow("Frame", frame)
        cv2.imshow("Processed Frame", frame_)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the test_frame_preprocessor function
test_frame_preprocessor()
