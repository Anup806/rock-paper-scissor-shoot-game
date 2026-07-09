"""
Session 3: Full Rock-Paper-Scissors Game
-----------------------------------------
This program is the complete game. The computer picks Rock, Paper, or
Scissors randomly, your webcam reads your hand gesture, and the program
tells you who won. It also keeps score.

How to run:
1. Open this file in VS Code.
2. Make sure opencv-python and mediapipe are installed:
   pip install opencv-python mediapipe
3. Click the Run button (or type: python session3_full_game.py)
4. Show your hand gesture, then press SPACE to lock in your move.
5. Press "q" on your keyboard to quit.
"""

import cv2
import mediapipe as mp
import random
import urllib.request
from pathlib import Path

from mediapipe_compat import patch_mediapipe_loader
from hand_drawing import draw_hand_landmarks

patch_mediapipe_loader()

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as base_options_lib

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def ensure_hand_landmarker_model():
    if MODEL_PATH.exists():
        return

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the hand landmarker model. "
            "Check your internet connection and try again."
        ) from exc


def create_hand_landmarker():
    ensure_hand_landmarker_model()
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options_lib.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


hand_landmarker = create_hand_landmarker()

cap = cv2.VideoCapture(0)

player_score = 0
computer_score = 0
result_text = "Show your hand, then press SPACE to play!"


def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]
    fingers = []
    for tip, joint in zip(tips, joints):
        if hand_landmarks[tip].y < hand_landmarks[joint].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def detect_gesture(fingers):
    if fingers == [0, 0, 0, 0]:
        return "Rock"
    elif fingers == [1, 1, 1, 1]:
        return "Paper"
    elif fingers == [1, 1, 0, 0]:
        return "Scissors"
    else:
        return "Unknown"


def decide_winner(player, computer):
    if player == computer:
        return "Tie!"
    wins = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
    if wins[player] == computer:
        return "You Win!"
    return "Computer Wins!"


current_gesture = "No hand detected"

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks, mp_vision.HandLandmarksConnections.HAND_CONNECTIONS)
            fingers = fingers_up(hand_landmarks)
            current_gesture = detect_gesture(fingers)
    else:
        current_gesture = "No hand detected"

    # Show live gesture, score, and last result
    cv2.putText(frame, f"Your move: {current_gesture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f"Score - You: {player_score}  Computer: {computer_score}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, result_text, (20, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Press SPACE to play, Q to quit", (20, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Session 3 - Rock Paper Scissors", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if current_gesture in ["Rock", "Paper", "Scissors"]:
            computer_choice = random.choice(["Rock", "Paper", "Scissors"])
            outcome = decide_winner(current_gesture, computer_choice)
            result_text = f"You: {current_gesture} vs Computer: {computer_choice} -> {outcome}"
            if outcome == "You Win!":
                player_score += 1
            elif outcome == "Computer Wins!":
                computer_score += 1
        else:
            result_text = "Show Rock, Paper, or Scissors clearly, then press SPACE."

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()