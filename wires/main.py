import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_opening


def neighbours2(y, x):
    return (y, x - 1), (y - 1, x)


def exist(B, nbs):
    nbs1 = []
    for i in nbs:
        if (i[0] >= 0 and i[0] < B.shape[0] and i[1] >= 0 and i[1] < B.shape[1]):
            if B[i] == 0:
                i = None
        else:
            i = None
        nbs1.append(i)
    return nbs1[0], nbs1[1]


def find(label, linked):
    j = label
    while linked[j] != 0:
        j = linked[j]
    return j


def union(label1, label2, linked):
    j = find(label1, linked)
    k = find(label2, linked)
    if j != k:
        linked[k] = j


def two_pass(B):
    LB = np.zeros_like(B)
    linked = np.zeros(B.size // 2 + 1, dtype="uint")
    label = 1
    for y in range(LB.shape[0]):
        for x in range(LB.shape[1]):
            if B[y, x] != 0:
                nbs = neighbours2(y, x)
                existed = exist(B, nbs)
                if existed[0] is None and existed[1] is None:
                    m = label
                    label += 1
                else:
                    lbs = [LB[n] for n in existed if n is not None]
                    m = min(lbs)
                LB[y, x] = m
                for n in existed:
                    if n is not None:
                        lb = LB[n]
                        if lb != m:
                            union(m, lb, linked)

    for y in range(LB.shape[0]):
        for x in range(LB.shape[1]):
            if B[y, x] != 0:
                new_label = find(LB[y, x], linked)
                if new_label != LB[y, x]:
                    LB[y, x] = new_label

    unique_labels = np.unique(LB)
    unique_labels = unique_labels[unique_labels != 0]
    for i, label in enumerate(unique_labels):
        LB[LB == label] = i + 1

    return LB


struct = np.ones((3, 2))
for i in [1, 2, 3, 4, 5, 6]:
    image = np.load(f"wires{i}.npy").astype(int)
    t_image = two_pass(image)
    izm_image = binary_opening(t_image, struct).astype(int)
    t_izm_image = two_pass(izm_image)

    total_wires = t_image.max()
    broken_parts_count = {}

    total_wires = t_image.max()
    broken_wires = 0
    print(f"На изображении wires{i}.npy: ")
    for pr in range(1, total_wires + 1):
        pr1 = (t_image == pr)
        izm_image = binary_opening(pr1, struct).astype(int)
        broken_count = np.max(two_pass(izm_image))


        if broken_count == 0:
            print("Провод не исправен")
        else:
            print(f"Провод: {broken_count} частей")



    unique_parts = np.unique(t_izm_image)
    unique_parts_count = len(unique_parts) - 1

    broken_parts_count[pr] = unique_parts_count

    total_broken_parts = sum(broken_parts_count.values())


    plt.subplot(121)
    plt.imshow(t_image)
    plt.title(f"Исходные провода - wires{i}.npy")

    plt.subplot(122)
    plt.imshow(t_izm_image)
    plt.title(f"После обработки - wires{i}.npy")

    plt.show()
