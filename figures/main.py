import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (binary_dilation,
                           binary_erosion,
                           binary_closing,
                           binary_opening)

#Функция получения пикселей соседей
def neighbours2(y, x):
    return ((y, x - 1), (y - 1, x))

#Функция получения пикселей соседей и являются ли они частью фигуры
def exist(B, nbs):
    left, top = nbs
    if left[1] >= 0 and left[1] < B.shape[1]:
        if B[left] == 0:
            left = None
    else:
        left = None
    if top[0] >= 0 and top[0] < B.shape[0]:
        if B[top] == 0:
            top = None
    else:
        top = None
    return left, top

#Ищет метку
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

#Двухпроходный алгоритм маркировки
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
    for fig, i in enumerate(np.unique(LB)):
        LB[LB == i] = fig
    return LB

#морфологическое открытие для удаления определенного вида фигур
def remove_shapes(labeled_img, shape_type):
    struct_element = np.ones((3, 3))
    shapes_removed = binary_opening(labeled_img, struct_element).astype("u8")
    print(f"Количество фигур типа {shape_type}:", two_pass(shapes_removed).max())
    return labeled_img - shapes_removed

#расширение фигур
def expand_objects(img, structure):
    eroded_img = binary_erosion(img, structure).astype("u8")
    expanded_img = img | eroded_img
    return expanded_img

#Морфологическое закрытие для сглаживания фигуры
def close_shapes(img, structure):
    return binary_closing(img, structure).astype("u8")


image = np.load("ps.npy").astype("u8")
print("Всего фигур:", two_pass(image).max())
plt.subplot(231)
plt.imshow(image)

# Прямоугольники
image = remove_shapes(image, "прямоугольник")
shape = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0]])
plt.subplot(232)
plt.imshow(shape)

# Подкова вверх
struct = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 1],
                   [1, 1, 0, 0, 1],
                   [1, 1, 1, 1, 1]])
ex_image = expand_objects(image, struct)
struct = np.array([[0, 0, 0],
                   [1, 1, 0],
                   [0, 0, 0]])
closed_shapes = close_shapes(ex_image, struct)
image = remove_shapes(closed_shapes, "подкова вверх")
shape = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0]])
plt.subplot(233)
plt.imshow(shape)

# Подкова вниз
struct = np.array([[1, 1, 1, 1, 1],
                   [1, 0, 0, 1, 1],
                   [1, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])
ex_image = expand_objects(image, struct)
struct = np.array([[0, 0, 0],
                   [1, 1, 0],
                   [0, 0, 0]])
closed_shapes = close_shapes(ex_image, struct)
image = remove_shapes(closed_shapes, "подкова вниз")
shape = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [0, 1, 1, 0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0]])
plt.subplot(234)
plt.imshow(shape)

# Подкова вправо
struct = np.array([[1, 1, 1, 1, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0, 0]])
ex_image = expand_objects(image, struct)
struct = np.array([[0, 0, 0],
                   [1, 1, 0],
                   [0, 0, 0]])
closed_shapes = close_shapes(ex_image, struct)
image = remove_shapes(closed_shapes, "подкова вправо")
shape = np.array([[0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 0],
                   [0, 1, 1, 0, 0, 0],
                   [0, 1, 1, 0, 0, 0],
                   [0, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0]])
plt.subplot(235)
plt.imshow(shape)

# Подкова влево
struct = np.array([[0, 0, 0, 1, 1, 1, 1],
                   [0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 1, 1]])
ex_image = expand_objects(image, struct)
struct = np.array([[0, 0, 0],
                   [0, 1, 1],
                   [0, 0, 0]])
closed_shapes = close_shapes(ex_image, struct)
image = remove_shapes(closed_shapes, "подкова влево")
shape = np.array([[0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 0],
                   [0, 0, 0, 1, 1, 0],
                   [0, 0, 0, 1, 1, 0],
                   [0, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0]])
plt.subplot(236)
plt.imshow(shape)
plt.show()


