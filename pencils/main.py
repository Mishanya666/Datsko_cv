import cv2
import numpy as np

total_pencils = 0

for img_index in range(1, 13):
    img_path = f"./images/images/img ({img_index}).jpg"
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Unable to load image {img_path}")
        continue
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = (0, 100, 0)
    upper_bound = (255, 255, 255)
    pencil_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    pencil_mask = cv2.dilate(pencil_mask, None, iterations=2)
    contours, _ = cv2.findContours(pencil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pencil_ratios = []
    for contour in contours:
        (center_x, center_y), (height, width), angle = cv2.minAreaRect(contour)
        max_side = max(height, width)
        min_side = min(height, width)
        if max_side > 1000 and min_side > 60:
            pencil_ratios.append(max_side / min_side)
        else:
            pencil_ratios.append(0)
    pencil_ratios = np.array(pencil_ratios)
    pencils_in_image = np.sum(pencil_ratios > 18)
    total_pencils += pencils_in_image
    print(f"Image {img_index}: {pencils_in_image} pencils")

print(f"Total pencils on all images: {total_pencils}")
