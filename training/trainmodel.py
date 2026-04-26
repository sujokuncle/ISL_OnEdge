"""
trainmodel.py
-------------
Trains an improved MLP for ISL alphabet recognition.

Key improvements over previous version:
  • Residual connections (skip layers) — prevents vanishing gradients
  • Class-weighted loss — fixes imbalanced dataset
  • Label smoothing — prevents overconfidence
  • Oversampling minority classes — explicit balance
  • Works with both 196-feature (rich) and 126-feature (plain) vectors
"""

import os
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils   import resample

import tensorflow as tf
from keras.utils    import to_categorical
from keras.models   import Model
from keras.layers   import (Dense, Dropout, BatchNormalization,
                             Input, Add, Activation)
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.losses    import CategoricalCrossentropy
from keras.optimizers import Adam

from config import (
    DATA_PATH, LOG_DIR, MODEL_JSON, MODEL_H5, MODEL_TFLITE,
    ACTIONS, EFFECTIVE_FEATURE,
)


# ─────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────

def load_dataset():
    sequences, labels = [], []
    label_map = {lbl: i for i, lbl in enumerate(ACTIONS)}
    skipped   = 0

    for action in ACTIONS:
        adir = os.path.join(DATA_PATH, action)
        if not os.path.exists(adir):
            print(f"  ⚠️  Missing: {action}"); continue

        folders = sorted(
            [d for d in os.listdir(adir) if os.path.isdir(os.path.join(adir,d))],
            key=lambda x: int(x)
        )
        for folder in folders:
            npy = os.path.join(adir, folder, "0.npy")
            if not os.path.exists(npy):
                skipped += 1; continue
            kp = np.load(npy)
            if kp.shape[0] != EFFECTIVE_FEATURE:
                skipped += 1; continue
            sequences.append(kp)
            labels.append(label_map[action])

    if skipped:
        print(f"  ℹ️  {skipped} files skipped (wrong shape or missing)")

    return np.array(sequences, dtype=np.float32), np.array(labels)


def oversample_minority(X, y, target_ratio=0.7):
    """
    Oversample classes below target_ratio × max_class_count.
    Uses random duplication with small noise.
    """
    counts = Counter(y)
    max_c  = max(counts.values())
    target = int(max_c * target_ratio)

    X_out, y_out = list(X), list(y)

    for cls, cnt in counts.items():
        if cnt < target:
            need    = target - cnt
            indices = np.where(y == cls)[0]
            chosen  = np.random.choice(indices, size=need, replace=True)
            noise   = np.random.normal(0, 0.005, (need, X.shape[1])).astype(np.float32)
            X_out.extend(X[chosen] + noise)
            y_out.extend([cls] * need)

    X_out = np.array(X_out, dtype=np.float32)
    y_out = np.array(y_out)
    perm  = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ─────────────────────────────────────────────
# MODEL — Residual MLP
# ─────────────────────────────────────────────

def residual_block(x, units, dropout_rate):
    """Dense → BN → ReLU → Dropout + skip connection."""
    shortcut = x
    out = Dense(units)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)

    # Match dimensions for skip connection if needed
    if shortcut.shape[-1] != units:
        shortcut = Dense(units)(shortcut)

    return Add()([out, shortcut])


def build_model(input_dim: int, num_classes: int) -> Model:
    """
    Residual MLP — better gradient flow than plain stack.
    Lightweight enough for Raspberry Pi after TFLite conversion.

    Architecture:
      Input(196) → Dense(512) → ResBlock(512) → ResBlock(256)
                 → ResBlock(128) → Dense(64) → Softmax(26)
    """
    inp = Input(shape=(input_dim,))

    x = Dense(512, activation='relu')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = residual_block(x, 512, 0.35)
    x = residual_block(x, 256, 0.30)
    x = residual_block(x, 128, 0.25)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        # Label smoothing: prevents overconfident wrong predictions
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['categorical_accuracy'],
    )
    return model


# ─────────────────────────────────────────────
# TFLITE EXPORT
# ─────────────────────────────────────────────

def export_tflite(model, X_sample):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_data():
        for i in range(min(200, len(X_sample))):
            yield [X_sample[i:i+1]]
    converter.representative_dataset = rep_data

    tflite_model = converter.convert()
    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)
    kb = os.path.getsize(MODEL_TFLITE) / 1024
    print(f"✅  TFLite saved → {MODEL_TFLITE}  ({kb:.1f} KB)")


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train():
    os.makedirs(LOG_DIR, exist_ok=True)

    print("📦  Loading dataset …")
    X, y = load_dataset()
    if len(X) == 0:
        print("❌  No data. Run collectdata.py first."); return

    print(f"    Raw samples  : {len(X)}")
    print(f"    Feature size : {EFFECTIVE_FEATURE}")
    print(f"    Classes      : {len(ACTIONS)}")

    # ── Class balance report ─────────────────
    counts = Counter(y)
    print("\n📊  Samples per class (before balancing):")
    for action in ACTIONS:
        idx = np.where(ACTIONS == action)[0][0]
        c   = counts.get(idx, 0)
        bar = "█" * (c // 100)
        print(f"    {action}: {c:5d}  {bar}")

    # ── Oversample minority classes ──────────
    print("\n⚖️   Oversampling minority classes …")
    X_bal, y_bal = oversample_minority(X, y, target_ratio=0.8)
    print(f"    Balanced samples: {len(X_bal)}")

    # ── Train / test split ───────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.20,
        stratify=y_bal, random_state=42,
    )
    y_train_cat = to_categorical(y_train, num_classes=len(ACTIONS))
    y_test_cat  = to_categorical(y_test,  num_classes=len(ACTIONS))

    print(f"    Train: {len(X_train)} | Test: {len(X_test)}\n")

    # ── Class weights (second safety net) ────
    from sklearn.utils.class_weight import compute_class_weight
    cw_arr  = compute_class_weight('balanced',
                                   classes=np.unique(y_train),
                                   y=y_train)
    cw_dict = {i: float(w) for i, w in enumerate(cw_arr)}

    # ── Build model ──────────────────────────
    model = build_model(EFFECTIVE_FEATURE, len(ACTIONS))
    model.summary()

    callbacks = [
        TensorBoard(log_dir=LOG_DIR),
        EarlyStopping(monitor='val_categorical_accuracy',
                      patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, min_lr=1e-6, verbose=1),
    ]

    model.fit(
        X_train, y_train_cat,
        epochs=200,
        batch_size=64,
        validation_data=(X_test, y_test_cat),
        class_weight=cw_dict,
        callbacks=callbacks,
    )

    # ── Evaluation ───────────────────────────
    print("\n📈  Test set evaluation:")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    present = sorted(set(y_test))
    names   = [ACTIONS[i] for i in present]
    print(classification_report(y_test, y_pred, labels=present, target_names=names))

    # ── Save Keras ───────────────────────────
    with open(MODEL_JSON, "w") as f:
        f.write(model.to_json())
    model.save_weights(MODEL_H5)
    print(f"✅  Keras model saved.")

    # ── Export TFLite ─────────────────────────
    try:
        export_tflite(model, X_test)
    except Exception as e:
        print(f"⚠️  TFLite export failed: {e}")
        print("    Run convert_to_tflite.py separately.")


if __name__ == "__main__":
    train()