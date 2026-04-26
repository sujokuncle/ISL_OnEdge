"""utils.py — FPS, overlays, logging"""

import cv2, time, os
import numpy as np
from config import ACTIONS, BAR_COLOR, TEXT_COLOR, ENABLE_LOGGING, DATA_PATH


class FPSCounter:
    def __init__(self, avg_over=30):
        self._times, self._avg, self._prev = [], avg_over, time.time()

    def tick(self):
        now = time.time()
        self._times.append(now - self._prev)
        self._prev = now
        if len(self._times) > self._avg: self._times.pop(0)
        avg = sum(self._times) / len(self._times)
        return 1.0 / avg if avg > 0 else 0.0


def draw_fps(image, fps):
    h, w = image.shape[:2]
    txt = f"FPS:{fps:.0f}"
    (tw,_),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(image, txt, (w-tw-8, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2, cv2.LINE_AA)


def draw_prediction_bar(image, probs, top_n=5):
    if probs is None: return
    h, w = image.shape[:2]
    for rank, idx in enumerate(np.argsort(probs)[::-1][:top_n]):
        prob  = probs[idx]
        bar_w = int(prob * 160)
        y0    = h - 32 - rank * 36
        y1    = h - 12 - rank * 36
        cv2.rectangle(image, (0,y0), (bar_w,y1), BAR_COLOR, -1)
        cv2.putText(image, f"{ACTIONS[idx]}:{prob:.2f}",
                    (bar_w+4, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_instructions(image):
    h, w = image.shape[:2]
    cv2.putText(image, "Q = quit", (8, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)


class DataLogger:
    def __init__(self):
        self._on = ENABLE_LOGGING
        if self._on: print("⚠️  DataLogger ENABLED")

    def log(self, action, kp, seq_id, frame_id=0):
        if not self._on: return
        d = os.path.join(DATA_PATH, action, str(seq_id))
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, f"{frame_id}.npy"), kp)