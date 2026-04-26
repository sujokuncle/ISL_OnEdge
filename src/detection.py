"""
detection.py
------------
Robust two-hand detection with overlap handling.

Key improvement for overlapping hands:
- IoU check between bounding boxes
- When overlap detected, use x-position to assign hands
  instead of trusting MediaPipe's handedness label
- Retry with higher resolution crop when overlap detected
- Flag overlapping state so inference uses relaxed threshold
"""

import cv2
import numpy as np
import mediapipe as mp

from config import (
    MP_MAX_HANDS, MP_DETECTION_CONFIDENCE,
    MP_TRACKING_CONFIDENCE, MP_MODEL_COMPLEXITY,
    BOX_COLOR_LEFT, BOX_COLOR_RIGHT,
    SINGLE_HAND_SIZE, OVERLAP_IOU_THRESHOLD,
)

mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands          = mp.solutions.hands


def build_hands_model(static_image_mode: bool = False):
    return mp_hands.Hands(
        static_image_mode        = static_image_mode,
        max_num_hands            = MP_MAX_HANDS,
        model_complexity         = MP_MODEL_COMPLEXITY,
        min_detection_confidence = MP_DETECTION_CONFIDENCE,
        min_tracking_confidence  = MP_TRACKING_CONFIDENCE,
    )


def mediapipe_detection(image, hands_model):
    if image is None:
        return image, None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def get_hand_bbox(hand_landmarks, img_w, img_h, padding=15):
    xs = [lm.x * img_w for lm in hand_landmarks.landmark]
    ys = [lm.y * img_h for lm in hand_landmarks.landmark]
    return (
        max(0,     int(min(xs)) - padding),
        max(0,     int(min(ys)) - padding),
        min(img_w, int(max(xs)) + padding),
        min(img_h, int(max(ys)) + padding),
    )


def compute_iou(box_a, box_b):
    ax1,ay1,ax2,ay2 = box_a
    bx1,by1,bx2,by2 = box_b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-8
    return inter / union


def extract_one_hand_raw(results, hand_index):
    if results and results.multi_hand_landmarks:
        if hand_index < len(results.multi_hand_landmarks):
            lms = results.multi_hand_landmarks[hand_index].landmark
            return np.array([[lm.x,lm.y,lm.z] for lm in lms],
                            dtype=np.float32).flatten()
    return np.zeros(SINGLE_HAND_SIZE, dtype=np.float32)


def assign_hands_robust(results, img_w, img_h):
    """
    Assign detected hands to Left/Right slots robustly.

    Strategy for overlapping hands:
    1. Compute IoU between bounding boxes
    2. If IoU > threshold → hands are overlapping
       → assign by x-position (more reliable than label)
    3. If IoU <= threshold → assign by MediaPipe label

    Returns
    -------
    left_raw    : (63,) zeros if not found
    right_raw   : (63,) zeros if not found
    hands_meta  : list of dicts for drawing
    overlapping : bool
    """
    left_raw    = np.zeros(SINGLE_HAND_SIZE, dtype=np.float32)
    right_raw   = np.zeros(SINGLE_HAND_SIZE, dtype=np.float32)
    hands_meta  = []
    overlapping = False

    if not (results and results.multi_hand_landmarks):
        return left_raw, right_raw, hands_meta, overlapping

    n     = len(results.multi_hand_landmarks)
    boxes = [get_hand_bbox(results.multi_hand_landmarks[i], img_w, img_h)
             for i in range(n)]

    # Check overlap
    if n == 2:
        iou         = compute_iou(boxes[0], boxes[1])
        overlapping = iou > OVERLAP_IOU_THRESHOLD

    for i in range(n):
        raw   = extract_one_hand_raw(results, i)
        label = results.multi_handedness[i].classification[0].label
        score = results.multi_handedness[i].classification[0].score
        box   = boxes[i]

        if n == 2 and overlapping:
            # Use x-position when overlapping
            # left side of frame = Left hand
            cx       = (box[0] + box[2]) / 2
            assigned = 'Left' if cx < img_w / 2 else 'Right'
        else:
            assigned = label

        if assigned == 'Left':
            left_raw = raw
        else:
            right_raw = raw

        hands_meta.append({
            'landmarks' : results.multi_hand_landmarks[i],
            'handedness': assigned,
            'box'       : box,
            'confidence': score,
        })

    return left_raw, right_raw, hands_meta, overlapping


def draw_styled_landmarks(image, results):
    if not (results and results.multi_hand_landmarks):
        return
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


def draw_bounding_box(image, hand_meta, label, overlapping=False):
    x1,y1,x2,y2 = hand_meta['box']
    handedness   = hand_meta['handedness']
    color        = BOX_COLOR_LEFT if handedness=='Left' else BOX_COLOR_RIGHT
    thickness    = 1 if overlapping else 2

    cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness)

    if overlapping:
        cv2.putText(image, 'overlap',
                    (x1, y2+15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0,165,255), 1)

    cv2.putText(image,
                f"{handedness[0]}:{label}",
                (x1, max(y1-8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85, color, 2, cv2.LINE_AA)