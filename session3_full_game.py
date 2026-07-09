"""
Session 3: Full Rock-Paper-Scissors Game (Round-Based, with Audio + Emoji)
---------------------------------------------------------------------------
Flow:
1. Press ENTER to start a round.
2. Stay silent for 1 second (no sound, no reveal yet).
3. Play "rock-paper-scissor-shoot.wav" once (~1 second).
4. The instant the audio finishes, the system reveals its random pick as
   an emoji (rock / paper / scissors), and your current hand gesture is
   captured and compared against it.
5. If you win: play "system-fad-denge.wav". If you lose: play the
   fah....wav sound. Ties and unclear gestures play no sound.
6. Result stays on screen for a few seconds, then it's ready for the next
   round. Press "q" anytime to quit.

Requirements (add to requirements.txt):
    opencv-python
    mediapipe==0.10.30
    Pillow                      (for full-color emoji rendering)

Note on audio: this uses Python's built-in `winsound` module (Windows only,
no pip install needed) instead of pygame, because pygame currently has no
prebuilt wheel for newer Python versions on Windows and fails to build from
source. winsound only plays .wav files, so make sure your three audio clips
in audio/ are .wav, not .mp3 (convert them once with any free mp3-to-wav
converter, or with `ffmpeg -i input.mp3 output.wav`).
"""

import time
import random
import winsound
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

from mediapipe_compat import patch_mediapipe_loader
from hand_drawing import draw_hand_landmarks

patch_mediapipe_loader()

from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as base_options_lib

import urllib.request

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


# ---------------------------------------------------------------------
# Hand gesture detection (unchanged from your working version)
# ---------------------------------------------------------------------

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


def decide_winner(player, system):
    """Standard rules: rock beats scissors, paper beats rock, scissors beats paper."""
    if player == system:
        return "Tie!"
    wins = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
    if wins[player] == system:
        return "You Win!"
    return "System Wins!"


# ---------------------------------------------------------------------
# Audio setup (stdlib winsound - Windows only, no pip install needed)
# ---------------------------------------------------------------------

AUDIO_DIR = Path(__file__).with_name("audio")


def load_sound_exact(filename):
    """Returns the path to a .wav file if it exists, else None."""
    path = AUDIO_DIR / filename
    if not path.exists():
        print(f"Warning: audio file not found: {path}")
        return None
    return str(path)


def load_sound_by_prefix(prefix):
    """Finds the first .wav in the audio folder starting with `prefix`.
    Used for the 'lose' sound since its filename has a long, easy-to-mistype
    run of repeated letters."""
    matches = sorted(AUDIO_DIR.glob(f"{prefix}*.wav"))
    if not matches:
        print(f"Warning: no audio file starting with '{prefix}' found in {AUDIO_DIR}")
        return None
    if len(matches) > 1:
        print(f"Warning: multiple files start with '{prefix}', using {matches[0].name}")
    return str(matches[0])


SHOOT_SOUND = load_sound_exact("rock-paper-scissor-shoot.wav")
WIN_SOUND = load_sound_exact("system-fad-denge.wav")
LOSE_SOUND = load_sound_by_prefix("fah")


def play_sound(path):
    """Plays a .wav file without blocking the game loop.
    SND_NODEFAULT stops Windows from silently substituting its own beep
    if the file can't be loaded - instead we catch and report the real
    error, so a bad file is obvious instead of confusing."""
    if path is None:
        return
    try:
        winsound.PlaySound(
            path,
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
    except RuntimeError as exc:
        print(f"Warning: failed to play '{path}': {exc}")
        print("This usually means the .wav isn't standard 16-bit PCM. "
              "Re-export it with: ffmpeg -i input.mp3 -ar 44100 -ac 2 -sample_fmt s16 output.wav")


# ---------------------------------------------------------------------
# Emoji rendering (PIL, so color emoji render correctly - cv2.putText
# cannot draw emoji glyphs)
# ---------------------------------------------------------------------

EMOJI_MAP = {"Rock": "🪨", "Paper": "📃", "Scissors": "✂️"}

EMOJI_FONT_PATH = "C:/Windows/Fonts/seguiemj.ttf"
try:
    emoji_font = ImageFont.truetype(EMOJI_FONT_PATH, 120)
    EMOJI_FONT_AVAILABLE = True
except Exception as exc:
    print(f"Warning: could not load emoji font ({exc}). Falling back to text labels.")
    EMOJI_FONT_AVAILABLE = False


def draw_emoji(frame, gesture_name, center_x, center_y):
    """Draws the emoji for a gesture ('Rock'/'Paper'/'Scissors') centered
    at (center_x, center_y). Falls back to a plain text label if the
    Windows emoji font isn't available on this machine."""
    emoji_char = EMOJI_MAP[gesture_name]

    if not EMOJI_FONT_AVAILABLE:
        cv2.putText(frame, gesture_name.upper(), (center_x - 90, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
        return frame

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), emoji_char, font=emoji_font, embedded_color=True)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    position = (center_x - text_w // 2, center_y - text_h // 2)
    draw.text(position, emoji_char, font=emoji_font, embedded_color=True)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------

COUNTDOWN_DURATION = 1.0       # silent pause after ENTER, before audio
AUDIO_DURATION = 1.0           # length of rock-paper-scissor-shoot.wav
RESULT_DISPLAY_DURATION = 3.0  # how long the result stays on screen

hand_landmarker = create_hand_landmarker()
cap = cv2.VideoCapture(0)

player_score = 0
system_score = 0

state = "IDLE"          # IDLE -> COUNTDOWN -> AUDIO -> RESULT -> IDLE
state_start_time = 0.0
system_choice = None
captured_gesture = None
result_text = ""

current_gesture = "No hand detected"

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read from webcam.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

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

    now = time.time()

    # ---- State machine ----
    if state == "IDLE":
        cv2.putText(frame, "Press ENTER to play a round", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    elif state == "COUNTDOWN":
        cv2.putText(frame, "Get ready...", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if now - state_start_time >= COUNTDOWN_DURATION:
            play_sound(SHOOT_SOUND)
            state = "AUDIO"
            state_start_time = now

    elif state == "AUDIO":
        cv2.putText(frame, "Rock, Paper, Scissors, Shoot!", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if now - state_start_time >= AUDIO_DURATION:
            # Reveal moment: lock in the system's random pick and whatever
            # gesture the player is showing right now.
            system_choice = random.choice(["Rock", "Paper", "Scissors"])
            captured_gesture = current_gesture

            if captured_gesture not in ["Rock", "Paper", "Scissors"]:
                result_text = "Couldn't see your hand clearly - try again!"
            else:
                outcome = decide_winner(captured_gesture, system_choice)
                result_text = f"You: {captured_gesture}  |  System: {system_choice}  ->  {outcome}"
                if outcome == "You Win!":
                    player_score += 1
                    play_sound(WIN_SOUND)
                elif outcome == "System Wins!":
                    system_score += 1
                    play_sound(LOSE_SOUND)
                # Tie -> no sound, by design

            state = "RESULT"
            state_start_time = now

    elif state == "RESULT":
        if system_choice is not None:
            frame = draw_emoji(frame, system_choice, w // 2, h // 2 - 30)
        cv2.putText(frame, result_text, (20, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        if now - state_start_time >= RESULT_DISPLAY_DURATION:
            state = "IDLE"

    # Score + quit hint always visible
    cv2.putText(frame, f"Score - You: {player_score}   System: {system_score}",
                (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, "ENTER to play, Q to quit", (20, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Session 3 - Rock Paper Scissors", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in (13, 10) and state == "IDLE":  # ENTER key starts a round
        state = "COUNTDOWN"
        state_start_time = now

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()