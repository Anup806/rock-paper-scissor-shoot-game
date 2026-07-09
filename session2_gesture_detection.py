"""
Session 2: Gesture Detection
----------------------------
This program builds on Session 1. It still tracks your hand, but now it
also guesses whether you are showing Rock, Paper, or Scissors, and prints
the guess on screen.

How to run:
1. Open this file in VS Code.
2. Make sure opencv-python and mediapipe are installed:
   pip install opencv-python mediapipe
3. Click the Run button (or type: python session2_gesture_detection.py)
4. Show Rock, Paper, or Scissors to the camera and watch the label change.
5. Press "q" on your keyboard to quit.
"""

import cv2
import mediapipe as mp
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


def fingers_up(hand_landmarks):
    """
    Returns a list of 4 numbers (1 = finger up, 0 = finger down)
    for the index, middle, ring, and pinky fingers.
    A finger counts as "up" if its tip is above the joint below it.
    """
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
    """Turns the finger pattern into a Rock / Paper / Scissors label."""
    if fingers == [0, 0, 0, 0]:
        return "Rock"
    elif fingers == [1, 1, 1, 1]:
        return "Paper"
    elif fingers == [1, 1, 0, 0]:
        return "Scissors"
    else:
        return "Unknown"


while True:
    success, frame = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_image)

    gesture_text = "No hand detected"

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks, mp_vision.HandLandmarksConnections.HAND_CONNECTIONS)
            fingers = fingers_up(hand_landmarks)
            gesture_text = detect_gesture(fingers)

    cv2.putText(frame, gesture_text, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Session 2 - Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()