"""
config.py — Central configuration for ISL Recognition System
"""
import os
import numpy as np

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, "MP_Data")
IMAGE_PATH   = os.path.join(BASE_DIR, "Images")
LOG_DIR      = os.path.join(BASE_DIR, "Logs")
MODEL_JSON   = os.path.join(BASE_DIR, "model.json")
MODEL_H5     = os.path.join(BASE_DIR, "model.h5")
MODEL_TFLITE = os.path.join(BASE_DIR, "model.tflite")

ACTIONS = np.array([
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
])
NUM_CLASSES = len(ACTIONS)

NO_SEQUENCES    = 400
SEQUENCE_LENGTH = 1
NUM_KEYPOINTS   = 21
SINGLE_HAND_SIZE = NUM_KEYPOINTS * 3        # 63  — raw xyz per hand
FEATURE_SIZE     = SINGLE_HAND_SIZE * 2     # 126 — left(63) + right(63)

# ── Rich feature size ────────────────────────
# Each hand: 63 xyz  +  20 bone-lengths  +  15 angles = 98
# Two hands: 98 × 2 = 196
RICH_SINGLE   = 63 + 20 + 15               # 98
RICH_FEATURE  = RICH_SINGLE * 2            # 196

# Set USE_RICH_FEATURES = True to use angle+distance features
# Set False to use plain normalised xyz (faster, simpler)
USE_RICH_FEATURES = True
EFFECTIVE_FEATURE = RICH_FEATURE if USE_RICH_FEATURES else FEATURE_SIZE

# ── MediaPipe ────────────────────────────────
MP_MAX_HANDS            = 2
MP_DETECTION_CONFIDENCE = 0.4
MP_TRACKING_CONFIDENCE  = 0.4
MP_MODEL_COMPLEXITY     = 0     # use 0 on Pi for speed

# ── Inference ────────────────────────────────
CONFIDENCE_THRESHOLD = 0.40
SMOOTHING_BUFFER     = 12       # frames for majority vote
OVERLAP_IOU_THRESHOLD = 0.30   # boxes above this are considered overlapping

# ── Display ──────────────────────────────────
BOX_COLOR_LEFT  = (255, 100,   0)
BOX_COLOR_RIGHT = (  0, 200, 100)
TEXT_COLOR      = (255, 255, 255)
BAR_COLOR       = (245, 117,  16)

ENABLE_LOGGING = False