import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from pathlib import Path

def extract_geometric_features(region):
    region_img = region.image
    height, width = region_img.shape
    area = region_img.sum() / region_img.size
    perimeter = region.perimeter / region_img.size
    aspect_ratio = height / width if width > 0 else 0
    cy, cx = region.local_centroid
    cy /= height
    cx /= width
    euler_number = region.euler_number

    central_region = region_img[int(0.45 * height):int(0.55 * height), int(0.45 * width):int(0.55 * width)]
    kl_feature = 3 * central_region.sum() / region_img.size if region_img.size > 0 else 0
    kls_feature = 2 * central_region.sum() / region_img.size if region_img.size > 0 else 0

    eccentricity = region.eccentricity * 8 if hasattr(region, 'eccentricity') else 0
    has_horizontal_lines = (np.mean(region_img, axis=0) > 0.87).sum() > 2
    has_vertical_lines = (np.mean(region_img, axis=1) > 0.85).sum() > 2
    medium_vertical_lines = (np.mean(region_img, axis=1) > 0.5).sum() > 2

    hole_area_ratio = region_img.sum() / region.filled_area if region.filled_area > 0 else 0
    solidity = region.solidity * 2 if hasattr(region, 'solidity') else 0
    return np.array([
        area, perimeter, cy, cx, euler_number, eccentricity,
        has_horizontal_lines * 3, hole_area_ratio, has_vertical_lines * 4,
        medium_vertical_lines * 5, kl_feature, aspect_ratio, kls_feature, solidity
    ])


def load_and_process_training_data(training_dir):
    training_dir = Path(training_dir)
    features, labels = [], []

    for label_idx, class_folder in enumerate(training_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        for img_path in class_folder.glob("*.png"):
            img = plt.imread(img_path)[:, :, :3].mean(axis=2)
            binary_img = (img > 0).astype(np.uint8)
            labeled_img = label(binary_img)
            regions = regionprops(labeled_img)
            if not regions:
                continue

            largest_region = max(regions, key=lambda r: r.area)
            features.append(extract_geometric_features(largest_region))
            labels.append(label_idx)
    return np.array(features), np.array(labels)

def train_knn_classifier(features, labels):
    knn = cv2.ml.KNearest_create()
    knn.train(features.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.float32).reshape(-1, 1))
    return knn

def classify_and_predict(knn_model, test_dir, class_names):
    test_dir = Path(test_dir)

    for img_path in test_dir.glob("*.png"):
        print(f"Picture {img_path.stem}: ", end=" ")
        img = plt.imread(img_path)[:, :, :3].mean(axis=2)
        binary_img = (img > 0.1).astype(np.uint8)
        labeled_img = label(binary_img)
        regions = regionprops(labeled_img)
        regions.sort(key=lambda r: r.centroid[1])
        prev_x = None
        for region in regions:
            if region.area <= 250:
                continue
            region_features = extract_geometric_features(region).astype(np.float32).reshape(1, -1)
            _, predictions, _, _ = knn_model.findNearest(region_features, k=3)
            bbox = region.bbox
            if prev_x is not None and bbox[1] - prev_x > 30:
                print(" ", end="")
            prev_x = bbox[3]
            print(class_names[int(predictions[0][0])][-1], end=" ")
        print()

if __name__ == "__main__":
    train_folder = "task/train/"
    test_folder = "task/"
    feature_matrix, label_list = load_and_process_training_data(train_folder)
    knn_classifier = train_knn_classifier(feature_matrix, label_list)
    class_names = [folder.name for folder in Path(train_folder).iterdir() if folder.is_dir()]
    classify_and_predict(knn_classifier, test_folder, class_names)

