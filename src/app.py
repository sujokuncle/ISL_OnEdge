"""
app.py
------
Real-time ISL Alphabet Recognition with Sound Output
  • Robust two-hand detection with IoU overlap handling
  • Rich feature extraction (xyz + bone lengths + angles)
  • Streak-locked majority-vote prediction smoothing
  • CLAHE webcam normalisation (domain gap fix)
  • Swap prediction fallback (fixes handedness issues)
  • Sound output via espeak (Pi) or pyttsx3 (laptop)
  • FPS display, probability bars, overlap warning
"""

import cv2
import sys
import numpy as np
import mediapipe as mp

from config     import EFFECTIVE_FEATURE, ACTIONS
from detection  import (mediapipe_detection, assign_hands_robust,
                        draw_styled_landmarks, draw_bounding_box,
                        build_hands_model)
from preprocessing import build_feature_vector
from inference  import GestureClassifier, PredictionSmoother
from utils      import (FPSCounter, draw_fps, draw_prediction_bar,
                        draw_instructions)
from sound_output import SoundOutput


# ─────────────────────────────────────────────
# SOUND SETTINGS
# ─────────────────────────────────────────────

# Minimum confidence to trigger sound
# Higher than display threshold — only speak when very sure
SPEECH_THRESHOLD = 0.75

# False = speak each letter individually
# True  = accumulate letters into words, speak on pause
WORD_BUILD_MODE  = False


# ─────────────────────────────────────────────
# WEBCAM NORMALISATION
# ─────────────────────────────────────────────

def normalize_webcam_frame(frame):
    """
    Apply CLAHE contrast enhancement to webcam frame.
    Closes the domain gap between training images
    (processed grayscale) and live webcam (color).
    Only enhances luminance — keeps color for MediaPipe.
    """
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l     = clahe.apply(l)
    lab   = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# DUAL PREDICTION WITH SWAP FALLBACK
# ─────────────────────────────────────────────

def predict_best(classifier, left_raw, right_raw, overlapping):
    """
    Run prediction twice:
      1. Normal:  [left | right]
      2. Swapped: [right | left]
    Return whichever gives higher confidence.

    Fixes letters like F where MediaPipe assigns
    the wrong hand to left/right slot due to mirroring.
    """
    feat1          = build_feature_vector(left_raw, right_raw)
    l1, conf1, p1 = classifier.predict(feat1, overlapping)

    both_present = (np.sum(np.abs(left_raw))  != 0 and
                    np.sum(np.abs(right_raw)) != 0)

    if both_present and (l1 == '?' or conf1 < 0.55):
        feat2          = build_feature_vector(right_raw, left_raw)
        l2, conf2, p2 = classifier.predict(feat2, overlapping)
        if conf2 > conf1:
            return l2, conf2, p2

    return l1, conf1, p1


# ─────────────────────────────────────────────
# HEADER OVERLAY
# ─────────────────────────────────────────────

def draw_header(image, label: str, overlapping: bool,
                confidence: float, word_buffer: str = ''):
    h, w = image.shape[:2]
    cv2.rectangle(image, (0,0), (w,50), (30,30,30), -1)

    if label == '?':
        text  = 'No gesture'
        color = (180,180,180)
    else:
        text  = f'ISL: {label}   ({confidence*100:.0f}%)'
        color = (0,255,0) if confidence > 0.75 else (0,200,255)

    cv2.putText(image, text, (12,36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                color, 2, cv2.LINE_AA)

    # Sound indicator — shown when speech will trigger
    if label != '?' and confidence >= SPEECH_THRESHOLD:
        cv2.putText(image, 'SOUND',
                    (w-80, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,255), 1, cv2.LINE_AA)

    # Word buffer display (word build mode only)
    if word_buffer:
        cv2.putText(image, f'Word: {word_buffer}',
                    (12,48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200,200,0), 1, cv2.LINE_AA)

    # Overlap warning
    if overlapping:
        cv2.putText(image, 'Hands overlapping',
                    (w-220, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0,165,255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────

print('🚀  Loading model ...')
try:
    classifier = GestureClassifier()
except FileNotFoundError as e:
    print(f'❌  {e}')
    sys.exit(1)

# Sound
print('🔊  Initialising sound ...')
sound = SoundOutput(
    confidence_threshold = SPEECH_THRESHOLD,
    same_letter_cooldown = 2.5,
    any_letter_cooldown  = 1.0,
    word_build_mode      = WORD_BUILD_MODE,
)
sound.start()

smoother    = PredictionSmoother()
fps_counter = FPSCounter()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('❌  Cannot open camera.')
    sound.stop()
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print('✅  Running. Press Q to quit.\n')
if WORD_BUILD_MODE:
    print('   C = clear word buffer')
    print('   S = speak word now\n')

last_probs = None
label      = '?'
last_conf  = 0.0


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

with build_hands_model(static_image_mode=False) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip + normalise
        frame = cv2.flip(frame, 1)
        frame = normalize_webcam_frame(frame)
        h, w  = frame.shape[:2]

        # ── Detect ────────────────────────────
        image, results = mediapipe_detection(frame, hands)

        # ── Assign hands with overlap detection
        left_raw, right_raw, hands_meta, overlapping = \
            assign_hands_robust(results, w, h)

        # ── Draw landmarks ────────────────────
        draw_styled_landmarks(image, results)

        # ── Feature vector + Predict ──────────
        features = build_feature_vector(left_raw, right_raw)

        if np.sum(np.abs(features)) > 0:
            raw_label, conf, probs = predict_best(
                classifier, left_raw, right_raw, overlapping
            )
            label      = smoother.update(raw_label)
            last_probs = probs
            last_conf  = conf

            # ── Trigger sound ──────────────────
            sound.notify(label=label, confidence=conf)

        else:
            smoother.reset()
            sound.notify(label='?', confidence=0.0)
            label     = '?'
            last_conf = 0.0

        # ── Draw bounding boxes ───────────────
        for hm in hands_meta:
            draw_bounding_box(image, hm, label, overlapping)

        # ── Overlays ──────────────────────────
        fps         = fps_counter.tick()
        word_buffer = sound.current_word if WORD_BUILD_MODE else ''

        draw_header(image, label, overlapping,
                    last_conf, word_buffer)
        draw_fps(image, fps)
        draw_prediction_bar(image, last_probs)
        draw_instructions(image)

        cv2.imshow('ISL Recognition', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and WORD_BUILD_MODE:
            sound.clear_word()
            print('Word buffer cleared')
        elif key == ord('s') and WORD_BUILD_MODE:
            if sound.current_word:
                word = sound.current_word
                sound.clear_word()
                sound._enqueue(word)
                print(f'Speaking word: {word}')

cap.release()
cv2.destroyAllWindows()
sound.stop()
print('👋  Done.')