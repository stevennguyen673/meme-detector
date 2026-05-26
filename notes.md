# Project Notes — Gesture Meme App

## Core Concepts

- Virtual environments isolate dependencies per project
- OpenCV is used for webcam + image processing
- A frame is just an image (numpy array)

---

# OpenCV

- cv2.imshow() displays the frame in a window
- cv2.waitKey(1) is required to keep the window responsive and update frames
- Without waitKey, the window freezes even if frames are still being processed

---

# Webcam Pipeline

- OpenCV accesses webcam through VideoCapture
- cap.read() continuously retrieves frames
- Each frame is a NumPy array representing image pixel data
- cv2.imshow() displays frames in a GUI window
- cv2.waitKey() keeps window responsive and handles keyboard input
- cap.release() frees webcam hardware resource

---

# TensorFlow Lite Pipeline

1. Webcam captures frame
2. OpenCV preprocesses image
3. TensorFlow Lite interpreter loads model
4. Input tensor is prepared
5. Model runs inference
6. Output tensor contains keypoints
7. Application logic reacts to pose

    webcam frame
        ↓
    resize (192x192)
        ↓
    cast to float32 (raw 0-255, do NOT normalize)
        ↓
    add batch dimension → shape (1, 192, 192, 3)
        ↓
    set tensor input
        ↓
    invoke model
        ↓
    get 17 keypoints

---

# Understanding Model Output Shape (MoveNet)

The model output has shape: (1, 1, 17, 3)

This is a 4D tensor:
- 1 → batch size (number of images processed at once)
- 1 → number of detected people (single-pose model always returns 1)
- 17 → number of keypoints (body joints like wrists, shoulders, etc.)
- 3 → values per keypoint: [y, x, confidence]

From a linear algebra perspective, this is a multi-dimensional matrix (tensor),
where each dimension adds structure to the data.

To make it easier to work with, we squeeze unnecessary dimensions:

    output[0][0] → shape becomes (17, 3)

This removes:
- batch dimension (since we only process 1 frame at a time)
- person dimension (since it's single-person detection)

After this transformation we are left with a 2D matrix:
17 rows (keypoints) × 3 columns (y, x, confidence)

---

# MoveNet Keypoint Index Map

0  - nose
1  - left eye
2  - right eye
3  - left ear
4  - right ear
5  - left shoulder
6  - right shoulder
7  - left elbow
8  - right elbow
9  - left wrist
10 - right wrist
11 - left hip
12 - right hip
13 - left knee
14 - right knee
15 - left ankle
16 - right ankle

Note: "left" and "right" are from the model's perspective.
If the webcam frame is flipped, swap left/right indices to match visual perspective.

---

# MoveNet Gotchas

- Output format is [y, x, confidence] NOT [x, y, confidence]
- Keypoint indices are from the model's perspective, not the camera's perspective
- Flipping the frame changes which side is visually "left" and "right"
- Low confidence keypoints (~0.05) can still pass loose thresholds and cause false triggers
- Wrists and ankles have naturally lower confidence than face keypoints
- This MoveNet variant expects raw 0-255 float32 input — do NOT normalize to 0-1

---

# State Management (main.py)

- current_meme: holds whichever meme image is active, None if no gesture detected
- prev_nose_x: stores nose x position from previous frame for movement detection

How prev_nose_x works:
- current_x = nose position this frame
- prev_nose_x = nose position last frame
- Comparing the two tells you which direction the head moved
- Only update prev_nose_x inside the confidence check to avoid storing bad values
- If confidence is low, keep the last known good position instead

---

# Gesture Logic (gestures.py)

- Pure math functions, no state
- Takes keypoints as input, returns True/False
- Keeping state out of gestures.py makes functions reusable and easier to test

head_move_left:
- Returns True if current_x - prev_x < -0.03
- x=0 is left side of screen, so decreasing x = moving left
- Threshold of 0.03 prevents tiny natural wobbles from triggering

left_hand_left_ear (visual left):
- Uses right wrist (10) and right ear (4) indices due to frame flip
- Calculates Euclidean distance between wrist and ear in normalized coordinates
- Also checks wrist is above shoulder to avoid false triggers at sides
- Confidence threshold of 0.3 for both points

---

# Architecture Overview

main.py
- Manages webcam connection
- Runs model inference every frame
- Tracks state (prev_nose_x, current_meme)
- Calls gesture functions and reacts to results

model.py
- Loads TFLite interpreter once at module level
- Exposes preprocess_frame() to format webcam frames for model input

gestures.py
- Pure logic functions
- No state, no imports from main or model
- Reusable and independently testable

---

# Troubleshooting Log

## Tensor Type Mismatch
- Symptom: "Got STRING but expected FLOAT32"
- Cause: preprocess_frame returning wrong type
- Fix: explicitly cast to np.float32, verify shape is (1, 192, 192, 3)

## Normalization Confusion
- Tried /255.0 and -1 to 1 normalization
- Confidence scores dropped to ~0.005
- Discovery: this MoveNet variant expects raw 0-255 float input, not normalized
- Lesson: always check input_details dtype before assuming normalization strategy

## Keypoint Axis Bug
- Used nose[0] for horizontal movement detection
- MoveNet outputs [y, x, confidence] so nose[0] is vertical axis
- Fix: use nose[1] for x-axis tracking

## Mirror/Keypoint Index Bug
- Gesture triggered on wrong side of body
- Cause: cv2.flip() mirrors frame but model keypoint indices stay fixed
- Fix: swap left/right keypoint indices to match visual perspective
- Lesson: also check all related indices (shoulder) not just the obvious ones

## prev_nose_x Never Updating
- Symptom: dog meme never triggered
- Cause: state variable was never written back at end of loop
- Fix: update prev_nose_x inside the confidence check block only
- Lesson: if a comparison variable never updates, the condition can never change

## False Trigger from Low Confidence Keypoints
- Symptom: angry birds meme triggered during head movement, not hand gesture
- Cause: wrist confidence threshold was 0.01, allowing ghost keypoints through
- Fix: raised threshold to 0.3 for all keypoints in gesture detection
- Lesson: low confidence detections are noise, not signal