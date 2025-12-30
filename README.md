# Gesture Recognition System

A real-time computer vision project that detects human gestures using MediaPipe
and displays a corresponding meme overlay when the gesture is held for a short duration.

## Features
- Real-time webcam processing
- Landmark-based gesture detection (no CNN training)
- Temporal filtering (gesture hold detection)
- Meme overlay using OpenCV

## Tech Stack
- Python
- OpenCV
- MediaPipe

## How it works
1. Webcam feed is captured using OpenCV
2. Pose/hand landmarks are extracted via MediaPipe
3. Gestures are detected using geometric rules
4. Corresponding meme is overlaid on the video feed

## Run
pip install -r mediapipe opencv-python
python main.py

## Note
- Mediapipe works better on python versions upto 3.11. So, create a environment as per it
