import numpy as np
from scipy.ndimage import binary_opening, binary_erosion


def get_neighbours(y, x):
    return [(y, x - 1), (y - 1, x)]


def valid_neighbours(binary_image, neighbours):
    valid = []
    for n in neighbours:
        if (0 <= n[0] < binary_image.shape[0] and
                0 <= n[1] < binary_image.shape[1] and
                binary_image[n] != 0):
            valid.append(n)
    return valid


def find_root(label, linked_labels):
    while linked_labels[label] != 0:
        label = linked_labels[label]
    return label


def union_labels(label1, label2, linked_labels):
    root1 = find_root(label1, linked_labels)
    root2 = find_root(label2, linked_labels)
    if root1 != root2:
        linked_labels[root2] = root1


def two_pass_labeling(binary_image):
    labeled_image = np.zeros_like(binary_image)
    linked_labels = np.zeros(binary_image.size // 2 + 1, dtype="uint")
    current_label = 1

    # Первый проход: присвоение меток
    for y in range(labeled_image.shape[0]):
        for x in range(labeled_image.shape[1]):
            if binary_image[y, x] != 0:
                neighbours = get_neighbours(y, x)
                existing_labels = valid_neighbours(binary_image, neighbours)

                if not existing_labels:
                    m = current_label
                    current_label += 1
                else:
                    labels = [labeled_image[n] for n in existing_labels]
                    m = min(labels)

                labeled_image[y, x] = m

                for n in existing_labels:
                    lb = labeled_image[n]
                    if lb != m:
                        union_labels(m, lb, linked_labels)

    # Второй проход: обновление меток
    for y in range(labeled_image.shape[0]):
        for x in range(labeled_image.shape[1]):
            if binary_image[y, x] != 0:
                new_label = find_root(labeled_image[y, x], linked_labels)
                labeled_image[y, x] = new_label

    # Переиндексация меток
    unique_labels = np.unique(labeled_image)
    unique_labels = unique_labels[unique_labels != 0]
    for i, label in enumerate(unique_labels):
        labeled_image[labeled_image == label] = i + 1

    return labeled_image

struct1 = np.array([[1, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1]])

struct2 = np.array([[0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0]])

image = np.load("stars.npy")

count_stars = two_pass_labeling(binary_erosion(image, struct1).astype(int))
count_pluses = two_pass_labeling(binary_opening(image, struct2).astype(int))

print('Stars: ', count_stars.max())
print('Pluses: ', count_pluses.max())
print('Total ammount: ', count_stars.max() + count_pluses.max())
