"""
Session 1: Hand Tracking
------------------------
This program turns on your webcam and draws dots and lines on your hand
using a pretrained AI model (MediaPipe). It does not understand gestures
yet - it only "sees" where your hand and fingers are.

How to run:
1. Open this file in VS Code.
2. Make sure opencv-python and mediapipe are installed:
   pip install -r requirements.txt
3. Click the Run button (or type: python session1_hand_tracking.py)
4. Press "q" on your keyboard to quit.
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

# Turn on the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)  # mirror image, feels natural
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            draw_hand_landmarks(frame, hand_landmarks, mp_vision.HandLandmarksConnections.HAND_CONNECTIONS)

    cv2.imshow("Session 1 - Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
