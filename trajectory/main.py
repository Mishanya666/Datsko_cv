import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from os import listdir
from os.path import isfile, join

data_path = "./motion/out/"

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

filenames = [f for f in listdir(data_path) if isfile(join(data_path, f))]
filenames.sort()

arr = np.load(join(data_path, filenames[0]))
labeled = label(arr)
regions = regionprops(labeled)

objects = []
for region in regions:
    cx, cy = region.centroid
    objects.append([[cx, cy]])

for file in filenames[1:]:
    frame = np.load(join(data_path, file))
    labeled = label(frame)
    regions = regionprops(labeled)

    # Матрица расстояний
    ds = []
    for j in range(len(regions)):
        ds.append([distance(regions[j].centroid, obj[-1]) for obj in objects])
    ds = np.array(ds)

    for i in range(len(objects)):
        nearest_idx = np.argmin(ds[:, i])
        objects[i].append(regions[nearest_idx].centroid)
        ds[nearest_idx, :] = np.inf

obs = np.array(objects)

for i, trajectory in enumerate(obs):
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"{i+1}")

plt.xlabel("X координаты")
plt.ylabel("Y координаты")
plt.legend()
plt.show()
