import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

directory = 'motion/out'

images = []

for i in range(100):
    filename = f'h_{i}.npy'
    file_path = os.path.join(directory, filename)
    img = np.load(file_path).astype(np.uint8)
    images.append(img)

trajectories = defaultdict(list)


def find_closest_trajectory(x, y, existing_trajectories):
    min_distance = float('inf')
    closest_id = -1

    for trajectory_id, points in existing_trajectories.items():
        if not points:
            continue
        last_point = points[-1]
        dist = np.sqrt((last_point[0] - x) ** 2 + (last_point[1] - y) ** 2)

        if dist < min_distance:
            min_distance = dist
            closest_id = trajectory_id

    return closest_id


initial_image = images[0]
cnts, _ = cv2.findContours(initial_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(cnts):
    (x, y), _ = cv2.minEnclosingCircle(cnt)
    trajectories[i].append((x, y))

for img in images[1:]:
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        (x, y), _ = cv2.minEnclosingCircle(cnt)

        closest_id = find_closest_trajectory(x, y, trajectories)

        trajectories[closest_id].append((x, y))

for trajectory in trajectories.values():
    trajectory_points = np.array(trajectory)
    plt.scatter(trajectory_points[:, 0], trajectory_points[:, 1], s=30, label='Trajectory')

plt.show()
