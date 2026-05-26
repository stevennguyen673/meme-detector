import numpy as np


# [x, y, confidence]

# angry bird meme

def left_hand_left_ear(keypoints):
    # keypoints are actually for right side of body but since cam is inverted have to use these
    left_wrist = keypoints[10]
    left_ear = keypoints[4]
    left_shoulder = keypoints[6]

    y1, x1, c1 = left_wrist
    y2, x2, c2 = left_ear
    sy, sx, sc = left_shoulder

    # relaxed confidence thresholds
    if c1 < 0.3 or c2 < 0.3:
        return False

    distance = np.sqrt(
        (x1 - x2)**2 +
        (y1 - y2)**2
    )

    # wrist must be above shoulder
    hand_raised = y1 < sy

    print("distance:", distance)

    return distance < 0.15 and hand_raised


# dog with hat meme
def head_move_left(prev_x, current_x):
    if prev_x is None:
        return False
    
    return (current_x - prev_x) < -0.03


    
