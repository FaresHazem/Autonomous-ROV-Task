import cv2
import numpy as np
# from Detection_and_Preprocessing_Block import FramePreprocessor
# from Decision_Making_Block import DecisionMaker, DistanceEstimator
from CameraHandler import VideoSaver, CameraHandler
import time

def calculate_execution_time(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time of {function.__name__}: {execution_time} seconds")
    return result


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def test_frame_preprocessor(camera):
    # preprocessor = FramePreprocessor()
    # dmaker = DecisionMaker()
    # estimator = DistanceEstimator(size_of_box=15.0, focal_length=1140, bbox=None)
    while True:
        timestamp, frame = camera.read()

        if frame is None:
            continue

        frame_ = cv2.resize(frame, (640, 480))
        # frame = hisEqulColor(frame_.copy())
        # preprocessor.frame = frame
        # bbox = calculate_execution_time(preprocessor.run)
        # if bbox is not None:
        #     x, y, w, h = bbox
        #     cv2.rectangle(frame, (int((x + 1) * 640 / 2), int((y + 1) * 480 / 2)),
        #                   (int((x + w + 2) * 640 / 2), int((y + h + 2) * 480 / 2)), 255, 2)

        # dmaker.bbox = bbox
        # estimator.bbox = preprocessor.bbox                    #un_normalized bbox
        # movement_vector_x, movement_vector_y = dmaker.run()
        # estimated_distance = estimator.Calculate()

        # print(f"Vector-X: {movement_vector_x} | Vector-Y: {movement_vector_y}")
        # print(f"Estimated Distance: {estimated_distance}")

        # cv2.arrowedLine(frame, (320, 240), (int(320 - movement_vector_x * 50), int(240 - movement_vector_y * 50)),
        #                 (0, 255, 0), 4)
        cv2.imshow("Frame", frame)
        # cv2.imshow("Processed Frame", frame_)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    choice =  "2" #input("Enter '1' for using a DVR camera or '2' for using a digital camera: ")
    capture = True #int(input("Do you want to capture a video: 0(No) | 1(yes): "))


    if choice == '1':
        camera_type = "DVR"
        camera_num = 1 #int(input("Enter DVR camera number/channel: "))
    elif choice == '2':
        camera_type = "Digital"
        camera_num = 1 #int(input("Enter digital camera number/channel: "))
    else:
        print("Invalid choice")
        exit()


    camera = CameraHandler(camera_type, camera_num)
    if(capture == True):
        video_name = input("Enter the name for saved video file: ")
        saver = VideoSaver(video_name + ".avi", frames=30)

    try:
        test_frame_preprocessor(camera)
    finally:
        saver.release()
