import zmq
import cv2
import numpy as np
from skimage.measure import label

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
port = 5555
socket.connect("tcp://192.168.0.100:%s" % port)
cv2.namedWindow("Client recv", cv2.WINDOW_GUI_NORMAL)

def count_objects(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([90, 50, 50])
    upper_green = np.array([150, 250, 250])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(img, img, mask=mask)



    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    _, binary = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_count = 0
    cube_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

            (x, y), rad = cv2.minEnclosingCircle(contour)
            center = int(x), int(y)
            rad = int(rad)
            cv2.circle(img, center, rad, (0, 255, 0), 2)

            if abs(cv2.contourArea(contour) - np.pi * rad * rad) < 1000:
                ball_count += 1
            else:
                width, height = rect[1]
                if abs(width - height) < 30:
                    cube_count += 1

    return ball_count, cube_count

while True:
    msg = socket.recv()
    frame = cv2.imdecode(np.frombuffer(msg, np.uint8), -1)
    ball_count, cube_count = count_objects(frame)
    cv2.putText(frame, f"Balls: {ball_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Cubes: {cube_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    cv2.imshow("Client recv", frame)

cv2.destroyAllWindows()
