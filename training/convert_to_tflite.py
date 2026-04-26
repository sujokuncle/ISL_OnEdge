"""
convert_to_tflite.py
--------------------
Standalone script to convert a trained Keras model to TFLite.
Run this on your laptop after training, then copy model.tflite
to the Raspberry Pi.

Usage
-----
  python convert_to_tflite.py
"""

import os
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

from config import MODEL_JSON, MODEL_H5, MODEL_TFLITE, DATA_PATH, ACTIONS, NO_SEQUENCES, FEATURE_SIZE


def representative_dataset():
    """
    Yields sample inputs for full-integer quantisation calibration.
    Loads real keypoint data if available, otherwise uses random values.
    """
    samples_yielded = 0
    for action in ACTIONS:
        for seq_idx in range(NO_SEQUENCES):
            npy = os.path.join(DATA_PATH, action, str(seq_idx), "0.npy")
            if os.path.exists(npy):
                kp = np.load(npy).reshape(1, FEATURE_SIZE).astype(np.float32)
                yield [kp]
                samples_yielded += 1
                if samples_yielded >= 200:
                    return

    # Fallback
    for _ in range(100):
        yield [np.random.rand(1, FEATURE_SIZE).astype(np.float32)]


def convert():
    print("📦  Loading Keras model …")
    with open(MODEL_JSON, "r") as f:
        model = model_from_json(f.read())
    model.load_weights(MODEL_H5)
    print("    Loaded successfully.")

    print("⚙️   Converting to TFLite (with INT8 quantisation) …")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(MODEL_TFLITE) / 1024
    print(f"✅  Saved → {MODEL_TFLITE}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    convert()
