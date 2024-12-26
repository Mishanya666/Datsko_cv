import cv2
import numpy as np

def count_color_objects(image, lower_bound, upper_bound):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    return len(filtered_contours)

yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])

red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])

red_lower2 = np.array([170, 100, 100])
red_upper2 = np.array([180, 255, 255])

capture = cv2.VideoCapture('output.avi')
frame_count = 0
match_count = 0

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame_count += 1

    # Подсчёт жёлтых объектов
    yellow_count = count_color_objects(frame, yellow_lower, yellow_upper)

    # Подсчёт красных объектов
    red_count1 = count_color_objects(frame, red_lower1, red_upper1)
    red_count2 = count_color_objects(frame, red_lower2, red_upper2)
    red_count = red_count1 + red_count2

    if yellow_count == 6 and red_count == 2:
        match_count += 1
        #cv2.imshow('Matched Frame', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

print(f"Число моих кадров: {match_count}")

capture.release()
cv2.destroyAllWindows()
