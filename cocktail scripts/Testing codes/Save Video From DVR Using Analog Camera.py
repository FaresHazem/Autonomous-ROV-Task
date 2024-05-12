import cv2
import numpy as np
import time
from threading import Thread
import os
DVR_LINK = '192.168.1.112'
USER_NAME = 'admin'
PASSWORD = 'admin000'
rtsp_link = lambda num: f"rtsp://{USER_NAME}:{PASSWORD}@{DVR_LINK}/h264/ch{num}/main/av_stream"
digital_link = lambda num: f"http://192.168.1.103:8081/?action=stream_{num}"
class VideoSaver:
    def __init__(self,video_name,frames=20,res=(640,480),fourcc=[*"XVID"]):
        fcc = cv2.VideoWriter_fourcc(*fourcc)
        self.out = cv2.VideoWriter(video_name,fcc,frames,res)

    def save_frame(self,frame):
        self.out.write(frame)
    
    def release(self):
        try:
            self.out.release()
        except:
            pass

class MjpgCamera:
    def __init__(self, num):
        self.stream = True
        self.num = num
        worked = False
        self.frame = None
        self.timestamp = time.time()
        while not worked:
            self.cap1 = cv2.VideoCapture(rtsp_link(num))
            worked, frame = self.cap1.read()
        print(f"Camera {num} initialized")
        self.t1 = Thread(target=self.__get_frame)
        self.t1.start()

    def __get_frame(self):
        while self.stream:
            frame = self.cap1.read()[1]
            self.frame = cv2.resize(frame, (640,480))
            self.timestamp = time.time()
    
    def read(self):
        return self.timestamp, self.frame

    def release(self):
        self.stream = False
        self.t1.join()
        print(f"Camera {self.num} joined")

index = 0
if __name__ == "__main__":
    right_camera = MjpgCamera(3)
    saver = VideoSaver("saved1.avi", 100, (640, 480))
    while True:
        t1, frame0 = right_camera.read()
        if frame0 is None:
            continue
        saver.save_frame(frame0)
        cv2.imshow("stacked", frame0)
        key = cv2.waitKey(1)
        if key == 27:
            break
    right_camera.release()
    saver.release()