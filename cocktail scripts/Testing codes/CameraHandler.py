
import cv2
import numpy as np
import time
from threading import Thread

####-----------------------HYPERPARAMETERS----------------------------------------------------------
DVR_LINK = '192.168.1.112'
USER_NAME = 'admin'
PASSWORD = 'admin000'
rtsp_link = lambda num: f"rtsp://{USER_NAME}:{PASSWORD}@{DVR_LINK}/h264/ch{num}/main/av_stream"
####------------------------------------------------------------------------------------------------

class VideoSaver:
    def __init__(self, video_name, frames=20, res=(640, 480), codec="XVID"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out = cv2.VideoWriter(video_name, fourcc, frames, res)

    def save_frame(self, frame):
        if frame is not None:
            self.out.write(frame)

    def release(self):
        self.out.release()

class CameraHandler:
    def __init__(self, camera_type, camera_num, resolution=(640, 480)):
        self.camera_type = camera_type
        self.camera_num = camera_num
        self.resolution = resolution
        self.stream = True
        self.frame = None
        self.timestamp = None

        if camera_type == "DVR":
            self.cap = cv2.VideoCapture(rtsp_link(camera_num))
        elif camera_type == "Digital":
            self.cap = cv2.VideoCapture(camera_num)
        else:
            raise ValueError("Unsupported camera type")

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera {camera_num}")

        self.thread = Thread(target=self.update_frame)
        self.thread.start()

    def update_frame(self):
        while self.stream:
            success, frame = self.cap.read()
            if success:
                self.frame = cv2.resize(frame, self.resolution)
                self.timestamp = time.time()

    def read(self):
        return self.timestamp, self.frame

    def release(self):
        self.stream = False
        self.thread.join()
        self.cap.release()
        print(f"Camera {self.camera_num} released")
