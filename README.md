# meme-detector

A real-time gesture recognition app that triggers memes based on your body movements, built with TensorFlow Lite and OpenCV.

## Demo

*GIF coming soon*

## How It Works

The app uses a pre-trained MoveNet pose estimation model to detect 17 body keypoints from your webcam feed in real time. Custom gesture logic interprets those keypoints and triggers a meme when a gesture is detected.

**Supported Gestures (Planning to add more):**
| Gesture | Meme |
|---|---|
| Left hand near left ear | Angry Birds |
| Head moves left | Dog with hat |

## Tech Stack

- Python
- TensorFlow Lite — on-device pose estimation inference
- MoveNet Thunder — pre-trained single-pose estimation model
- OpenCV — webcam capture and image display
- NumPy — keypoint math and array manipulation


1. Clone the repo
```bash
git clone https://github.com/stevennguyen673/meme-detector.git
cd meme-detector
```

2. Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install tensorflow opencv-python numpy
```

4. Run the app
```bash
cd src
python main.py
```

Press `q` to quit.

## Design Decisions

- **Separated gesture logic from application logic** — `gestures.py` contains pure functions with no state, making them independently testable and reusable
- **Module-level model loading** — the TFLite interpreter loads once at import time rather than every frame, keeping inference fast
- **Confidence thresholding** — keypoints below 0.3 confidence are ignored to prevent ghost detections from triggering gestures
- **State-gated updates** — `prev_nose_x` only updates when nose confidence is sufficient, preventing bad values from corrupting movement detection

## Web Version

A browser-based port of this project is available at:
- **[Live Demo](https://stevennguyen673.github.io/meme-detector-web)**
- **[Repository](https://github.com/stevennguyen673/meme-detector-web)**

Rebuilt using TensorFlow.js and the browser webcam API — no installation required.
