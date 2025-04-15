import ultralytics
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from skimage import draw
import time

rr, cc = draw.disk((5, 5), 5)
struct = np.zeros((11,11), dtype=np.uint8)
struct[rr, cc] = 1

path = Path(__file__).parent
model_path = path / "facial_best.pt"

oranges = cv2.imread("oranges.png")
hsv_oranges = cv2.cvtColor(oranges, cv2.COLOR_BGR2HSV)

lower = np.array([10, 240, 210])
upper = np.array([15, 255, 255])
mask = cv2.inRange(hsv_oranges, lower, upper)
mask = cv2.dilate(mask, struct)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=cv2.contourArea)
m = cv2.moments(sorted_contours[-1])
cx = int(m["m10"] / m["m00"])
cy = int(m["m01"] / m["m00"])
bbox = cv2.boundingRect(sorted_contours[-1])
x, y, w, h = bbox

model = YOLO(model_path)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка при открытии камеры")
    exit()

while True:
    ret, image = cap.read()
    if not ret:
        break

    result = model(image)[0]
    masks = result.masks

    if masks is None or len(masks) == 0:
        cv2.imshow("Webcam", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    annotated = result.plot()

    global_mask = masks[0].data.numpy()[0, :, :]
    for mask in masks[1:]:
        global_mask += mask.data.numpy()[0, :, :]

    global_mask = cv2.resize(global_mask, (image.shape[1], image.shape[0])).astype(np.uint8)
    global_mask = cv2.dilate(global_mask, struct)

    gglobal_mask = cv2.bitwise_and(image, image, mask=global_mask)

    pos = np.where(global_mask > 0)
    min_y, max_y = int(np.min(pos[0]) * 0.8), int(np.max(pos[0]) * 1.1)
    min_x, max_x = int(np.min(pos[1]) * 0.8), int(np.max(pos[1]) * 1.1)
    global_mask = global_mask[min_y:max_y, min_x:max_x]
    gglobal_mask = gglobal_mask[min_y:max_y, min_x:max_x]

    resized_parts = cv2.resize(gglobal_mask, (w, h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(global_mask, (w, h), interpolation=cv2.INTER_AREA) * 255

    orange_clone = oranges.copy()
    roi = orange_clone[y:y+h, x:x+w]
    bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(resized_mask))
    combined_oranges = cv2.add(bg, resized_parts)
    orange_clone[y:y+h, x:x+w] = combined_oranges

    cv2.VideoCapture()
    cv2.imshow("oranges", orange_clone)
    cv2.imshow("image", annotated)
    cv2.imshow("Mask", gglobal_mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
