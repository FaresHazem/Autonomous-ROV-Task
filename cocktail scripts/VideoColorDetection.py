import cv2
import numpy as np
import imutils

def nothing(x):
    pass

cap = cv2.VideoCapture(r"D:\College\External Courses\CrocoMarine ROV Competitions\Distance_Estimation_Using_YOLOv8\Data\Videos\Sample 5.mp4");
#cap = cv2.VideoCapture(0);
record = False
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

cout = 1
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

while True:
    #frame = cv2.imread('photo_2023-02-23_16-34-29.jpg')
    _, frame = cap.read()
    frame = imutils.resize(frame, height=480)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)
    erroded = cv2.erode(mask, (15,15), iterations= 2)
    dilaed = cv2.dilate(erroded, (25,25), iterations= 8)

    result = cv2.bitwise_and(frame, frame, mask=dilaed)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", dilaed)
    cv2.imshow("res", result)

    key = cv2.waitKey(50)
    if key == 27:
        break

    elif key == ord('s'):
        cv2.imwrite(f'frame_{cout}.png', frame)
        print(f'frame {cout} saved')
        cout += 1
        # press space key to start recording

    elif key % 256 == 32:
        record = True
        print(f'RECORDING {record}')

    elif record:
        out.write(frame)

cap.release()
cv2.destroyAllWindows()

