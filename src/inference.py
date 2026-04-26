"""
inference.py
------------
Model loading + robust prediction with multi-layer smoothing.

Smoothing pipeline:
  1. Confidence gate  — reject if below threshold
  2. Overlap guard    — lower threshold when hands overlap
  3. Majority vote    — deque of last N predictions
  4. Streak lock      — only emit label after K consecutive same predictions
"""

import os
import numpy as np
from collections import deque, Counter

from config import (
    MODEL_TFLITE, MODEL_JSON, MODEL_H5,
    ACTIONS, CONFIDENCE_THRESHOLD,
    SMOOTHING_BUFFER, EFFECTIVE_FEATURE,
)

OVERLAP_CONF_THRESHOLD = 0.50   # lower bar when hands overlap
STREAK_REQUIRED        = 3      # consecutive same predictions before lock


class GestureClassifier:

    def __init__(self):
        self._use_tflite = False
        self._interpreter = self._keras_model = None

        if os.path.exists(MODEL_TFLITE):
            self._load_tflite()
        elif os.path.exists(MODEL_H5) and os.path.exists(MODEL_JSON):
            self._load_keras()
        else:
            raise FileNotFoundError(
                f"No model found.\n"
                f"  Expected: {MODEL_TFLITE}  or  {MODEL_JSON}+{MODEL_H5}"
            )

    def _load_tflite(self):
        try:
            import tflite_runtime.interpreter as tflite
            Interp = tflite.Interpreter
        except ImportError:
            import tensorflow as tf
            Interp = tf.lite.Interpreter

        self._interpreter = Interp(model_path=MODEL_TFLITE)
        self._interpreter.allocate_tensors()
        self._in  = self._interpreter.get_input_details()
        self._out = self._interpreter.get_output_details()
        self._use_tflite = True
        print(f"✅  TFLite loaded → {MODEL_TFLITE}")

    def _load_keras(self):
        from keras.models import model_from_json
        with open(MODEL_JSON) as f:
            self._keras_model = model_from_json(f.read())
        self._keras_model.load_weights(MODEL_H5)
        print(f"✅  Keras loaded → {MODEL_H5}")

    def _run(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(1, EFFECTIVE_FEATURE).astype(np.float32)
        if self._use_tflite:
            self._interpreter.set_tensor(self._in[0]['index'], x)
            self._interpreter.invoke()
            return self._interpreter.get_tensor(self._out[0]['index'])[0]
        return self._keras_model.predict(x, verbose=0)[0]

    def predict(self, keypoints: np.ndarray,
                overlapping: bool = False):
        """
        Parameters
        ----------
        keypoints   : (EFFECTIVE_FEATURE,) combined feature vector
        overlapping : bool — if True, use relaxed confidence threshold

        Returns
        -------
        label      : str
        confidence : float
        probs      : np.ndarray (26,)
        """
        probs      = self._run(keypoints)
        idx        = int(np.argmax(probs))
        confidence = float(probs[idx])
        thresh     = OVERLAP_CONF_THRESHOLD if overlapping else CONFIDENCE_THRESHOLD
        label      = ACTIONS[idx] if confidence >= thresh else "?"
        return label, confidence, probs


class PredictionSmoother:
    """
    Three-layer smoothing:
      1. Majority vote over last SMOOTHING_BUFFER predictions
      2. Streak lock — only change output after STREAK_REQUIRED
         consecutive same raw predictions
      3. Returns "?" when buffer is insufficient
    """

    def __init__(self, buffer_size: int = SMOOTHING_BUFFER):
        self._buf      = deque(maxlen=buffer_size)
        self._streak   = 0
        self._last_raw = None
        self._locked   = "?"

    def update(self, raw_label: str) -> str:
        # Streak tracking
        if raw_label == self._last_raw:
            self._streak += 1
        else:
            self._streak   = 1
            self._last_raw = raw_label

        # Only add to buffer after streak requirement met
        if self._streak >= STREAK_REQUIRED:
            self._buf.append(raw_label)

        if not self._buf:
            return "?"

        majority, _ = Counter(self._buf).most_common(1)[0]
        return majority

    def reset(self):
        self._buf.clear()
        self._streak   = 0
        self._last_raw = None
        self._locked   = "?"