import cv2
import time
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

cv2.namedWindow("RPS Game", cv2.WINDOW_NORMAL)
camera = cv2.VideoCapture(0)

game_state = "idle"
start_time = 0
countdown = 0
player1_gesture = ""
player2_gesture = ""
result_text = ""

while camera.isOpened():
    success, frame = camera.read()
    if not success:
        break

    cv2.putText(frame, f"State: {game_state} | Time: {5 - countdown:.1f}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    detections = model(frame)
    prediction = detections[0]

    if not prediction:
        continue

    if len(prediction.boxes.xyxy) == 2:
        detected_labels = []

        for cls_id, coords in zip(prediction.boxes.cls, prediction.boxes.xyxy):
            x1, y1, x2, y2 = coords.numpy().astype(int)
            label = prediction.names[int(cls_id)].lower()
            detected_labels.append(label)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 3)
            cv2.putText(frame, label, (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        player1_gesture, player2_gesture = detected_labels

        if player1_gesture == "rock" and player2_gesture == "rock" and game_state == "idle":
            game_state = "countdown"
            start_time = time.time()

    if game_state == "countdown":
        countdown = round(time.time() - start_time, 1)

    if countdown >= 5 and game_state == "countdown":
        game_state = "result"
        countdown = 5

        if player1_gesture == player2_gesture:
            result_text = "It's a draw!"
        elif player1_gesture == "rock":
            result_text = "Player 1 wins!" if player2_gesture == "scissors" else "Player 2 wins!"
        elif player1_gesture == "paper":
            result_text = "Player 1 wins!" if player2_gesture == "rock" else "Player 2 wins!"
        elif player1_gesture == "scissors":
            result_text = "Player 1 wins!" if player2_gesture == "paper" else "Player 2 wins!"

    cv2.putText(frame, result_text, (200, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 100, 100), 2)

    cv2.imshow("RPS Game", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        game_state = "idle"
        countdown = 0
        result_text = ""
        player1_gesture = ""
        player2_gesture = ""

camera.release()
cv2.destroyAllWindows()
