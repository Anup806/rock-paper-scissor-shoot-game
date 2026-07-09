"""Small OpenCV helpers for drawing MediaPipe hand landmarks."""

from __future__ import annotations

import cv2


def draw_hand_landmarks(frame, hand_landmarks, connections):
    """Draw landmark points and connecting lines on a BGR frame."""

    if not hand_landmarks:
        return

    height, width = frame.shape[:2]

    for connection in connections:
        start = hand_landmarks[connection.start]
        end = hand_landmarks[connection.end]
        start_point = (int(start.x * width), int(start.y * height))
        end_point = (int(end.x * width), int(end.y * height))
        cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    for landmark in hand_landmarks:
        point = (int(landmark.x * width), int(landmark.y * height))
        cv2.circle(frame, point, 5, (0, 0, 255), -1)