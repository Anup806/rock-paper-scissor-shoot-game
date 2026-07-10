# Rock Paper Scissors Shoot — Hand Gesture Game

A webcam-based Rock-Paper-Scissors game built for Nepal Botworks' 3-session
introductory AI/CV workshop. Uses MediaPipe hand tracking to read your
gesture, plays audio cues, and shows a live win/lose/tie result.

Repo: https://github.com/Anup806/rock-paper-scissor-shoot-game

---

## Requirements

- Windows 10/11
- Python 3.10+ (project has been tested against recent Python versions;
  see the MediaPipe note below if you hit an import error)
- A working webcam
- Internet connection on first run only (to download the hand-tracking
  model file, `hand_landmarker.task`, if it isn't already present)

---

## Setup

1. **Clone the repo** and open a terminal in the project folder:
   ```powershell
   git clone https://github.com/Anup806/rock-paper-scissor-shoot-game.git
   cd rock-paper-scissor-shoot-game
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Check your audio files are `.wav`, not `.mp3`.** `winsound` (used for
   audio playback) only supports uncompressed WAV files. If your `audio/`
   folder still has `.mp3` files, convert them once:
   ```powershell
   ffmpeg -i "audio\rock-paper-scissor-shoot.mp3" -ar 44100 -ac 2 -sample_fmt s16 "audio\rock-paper-scissor-shoot.wav" -y
   ffmpeg -i "audio\system-fad-denge.mp3" -ar 44100 -ac 2 -sample_fmt s16 "audio\system-fad-denge.wav" -y
   ffmpeg -i "audio\fah....mp3" -ar 44100 -ac 2 -sample_fmt s16 "audio\fah....wav" -y
   ```
   No ffmpeg? Use a free converter like cloudconvert.com and make sure to
   export standard 16-bit PCM WAV, then re-test with `test_audio.py` below.

   You can verify your audio files play correctly on their own, before
   ever touching the webcam game, by running:
   ```powershell
   python test_audio.py
   ```

---

## Running the Game

```powershell
python session3_full_game.py
```

- A webcam window opens showing your hand being tracked.
- Press **ENTER** to start a round.
- Stay ready — after a short silent pause, you'll hear "Rock, Paper,
  Scissors, Shoot!"
- Right after the audio finishes, show your hand gesture within about a
  second — that's your captured move.
- The system's random pick appears as an emoji in a box in the top-right
  corner. **WIN** / **LOSE** / **TIE** appears in bold at the center of
  the screen based on the outcome.
- Press **ENTER** again for another round, or **Q** to quit at any time.

### Gestures Recognized

| Gesture  | Hand shape                          |
|----------|--------------------------------------|
| Rock     | Closed fist (✊)                      |
| Paper    | Open palm, all fingers extended (🖐️) |
| Scissors | Index + middle fingers extended (✌️) |

---

## Other Scripts in This Repo

| File | Purpose |
|---|---|
| `session1_hand_tracking.py` | Session 1 demo — just draws hand landmarks live, no gesture logic. |
| `session2_gesture_detection.py` | Session 2 demo — detects and labels Rock/Paper/Scissors on screen, no game/scoring. |
| `session3_full_game.py` | The full game described above. |
| `test_audio.py` | Standalone check that your 3 `.wav` files play correctly, independent of the webcam game. |
| `mediapipe_compat.py` | Compatibility shim for loading MediaPipe's hand landmarker model across versions. |
| `hand_drawing.py` | Draws the hand-landmark skeleton on the video frame. |

---

## Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
Recent MediaPipe releases (0.10.30+) removed the old `mp.solutions` API.
This project already uses the newer Tasks API (`HandLandmarker`) via
`mediapipe_compat.py`, so make sure your installed version matches what's
pinned in `requirements.txt` — don't hand-edit that version without
re-testing.

**Sound doesn't play, plays the wrong thing, or plays a generic beep**
Almost always a WAV format issue — `winsound` only supports standard
16-bit PCM WAV. Re-export with the `ffmpeg` command above and re-check
with `python test_audio.py`.

**Gesture is never detected / always says "Unclear"**
Make sure your hand is well-lit, fully in frame, and held steady during
the ~1 second capture window right after the "Shoot!" audio ends —
lighting and hand angle affect landmark accuracy.

**First run is slow or needs internet**
The first launch downloads `hand_landmarker.task` (the hand-tracking
model) if it isn't already in the project folder. After that, it's
reused locally and no internet is needed.

**Webcam window doesn't open / wrong camera used**
`cv2.VideoCapture(0)` opens the default camera. If you have multiple
cameras, try changing `0` to `1` in the relevant script.

---

## Credits

Built for Nepal Botworks by Anup, as part of a 3-session high-school
introductory workshop on programming, computer vision, and AI.