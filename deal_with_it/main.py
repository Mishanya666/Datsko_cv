import cv2
import numpy as np

lbp_face_cascade = "lbpcascades/lbpcascade_frontalface.xml"
haar_eye_cascade_1 = "haarcascades/haarcascade_eye.xml"
haar_eye_cascade_2 = "haarcascades/haarcascade_eye_tree_eyeglasses.xml"

eye_classifier_1 = cv2.CascadeClassifier(haar_eye_cascade_1)
eye_classifier_2 = cv2.CascadeClassifier(haar_eye_cascade_2)


def detect_objects(image, classifier, scale_factor=1.1, min_neighbors=5):
    result_image = image.copy()
    detected_objects = classifier.detectMultiScale(result_image, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    for (x, y, width, height) in detected_objects:
        cv2.rectangle(result_image, (x, y), (x + width, y + height), (0, 128, 255), 2)
    return result_image


def apply_glasses(glasses_image, face_image, classifier, scale_factor=1.1, min_neighbors=5):
    result_image = face_image.copy()
    detected_eyes = classifier.detectMultiScale(result_image, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    if len(detected_eyes) == 2:
        eye1_x, eye1_y, eye1_w, eye1_h = detected_eyes[0]
        eye2_x, eye2_y, eye2_w, eye2_h = detected_eyes[1]

        center_x = int((eye1_x + eye2_x + eye2_w) / 2)
        center_y = int((eye1_y + eye2_y + eye2_h) / 2)

        glasses_width = max(eye2_x + eye2_w - eye1_x, eye1_x + eye1_w - eye2_x)
        glasses_height = max(eye2_y + eye2_h - eye1_y, eye1_y + eye1_h - eye2_y)

        resized_glasses = cv2.resize(glasses_image, (glasses_width + 50, glasses_height))
        if resized_glasses.shape[-1] == 4:
            resized_glasses = resized_glasses[:, :, :3]

        for i in range(center_y - glasses_height // 2, center_y - glasses_height // 2 + resized_glasses.shape[0]):
            for j in range(center_x - glasses_width // 2 - 25,
                           center_x - glasses_width // 2 - 25 + resized_glasses.shape[1]):
                if np.all(resized_glasses[
                              i - (center_y - glasses_height // 2), j - (center_x - glasses_width // 2 - 25)] < (
                          250, 250, 250)):
                    result_image[i, j] = resized_glasses[
                        i - (center_y - glasses_height // 2), j - (center_x - glasses_width // 2 - 25)]

    return result_image


def main():
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    camera.set(cv2.CAP_PROP_EXPOSURE, -3)

    glasses = cv2.imread("dealwithit.png", cv2.IMREAD_UNCHANGED)
    gray_glasses = cv2.cvtColor(glasses, cv2.COLOR_BGR2GRAY)
    gray_glasses = 255 - gray_glasses
    contours, _ = cv2.findContours(gray_glasses, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    glasses = glasses[y:y + h, x:x + w]

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        output_frame = apply_glasses(glasses, frame, eye_classifier_2, 1.2, 5)

        cv2.imshow("Camera", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
