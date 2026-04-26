"""
collectdata.py
--------------
Extracts rich hand features from grayscale ISL images.
Applies data augmentation to fix class imbalance.

Feature vector: 196 (rich) or 126 (plain) per sample
Augmentation multiplier: ~12x per original image
"""

import os, cv2, glob
import numpy as np
import mediapipe as mp

from config import (
    IMAGE_PATH, DATA_PATH, ACTIONS, NO_SEQUENCES,
    SINGLE_HAND_SIZE, EFFECTIVE_FEATURE,
)
from detection     import mediapipe_detection, extract_one_hand_raw, assign_hands_robust
from preprocessing import build_feature_vector

mp_hands_sol = mp.solutions.hands


# ─────────────────────────────────────────────
# IMAGE ENHANCEMENT
# ─────────────────────────────────────────────

def enhance(frame):
    h, w = frame.shape[:2]
    if w < 500 or h < 500:
        s = max(500/w, 500/h)
        frame = cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_CUBIC)

    # Grayscale → colorised
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    gray  = clahe.apply(gray)
    bgr   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    b,g,r = cv2.split(bgr)
    bgr   = cv2.merge((b, cv2.add(g,15), cv2.add(r,30)))
    bgr   = cv2.GaussianBlur(bgr, (3,3), 0)

    # Replace black background
    gchk = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgr[gchk < 40] = [128,128,128]

    # Sharpen
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(bgr, -1, k)


# ─────────────────────────────────────────────
# AUGMENTATION  (landmark-space + image-space)
# ─────────────────────────────────────────────

def augment_images(frame):
    """Return list of augmented image variants."""
    h, w = frame.shape[:2]
    out = [frame]

    # Brightness
    for a, b in [(1.1,10),(0.9,-10),(1.2,20),(0.8,-20)]:
        out.append(cv2.convertScaleAbs(frame, alpha=a, beta=b))

    # Rotation ±5°, ±10°
    for angle in [-10,-5,5,10]:
        M   = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
        out.append(cv2.warpAffine(frame, M, (w,h),
                   borderMode=cv2.BORDER_CONSTANT, borderValue=[128,128,128]))

    # Zoom
    for sc in [0.92, 0.96]:
        nw,nh = int(w*sc), int(h*sc)
        x1,y1 = (w-nw)//2, (h-nh)//2
        crop  = frame[y1:y1+nh, x1:x1+nw]
        if crop.size > 0:
            out.append(cv2.resize(crop, (w,h)))

    # Noise
    noise = np.random.normal(0, 6, frame.shape).astype(np.int16)
    out.append(np.clip(frame.astype(np.int16)+noise, 0, 255).astype(np.uint8))

    return out   # ~12 variants


def augment_keypoints(left_raw, right_raw, n=3):
    """
    Keypoint-space augmentation: add tiny Gaussian noise to raw landmarks.
    Returns list of (left_raw, right_raw) tuples including the original.
    """
    pairs = [(left_raw.copy(), right_raw.copy())]
    for _ in range(n):
        lnoise = np.random.normal(0, 0.008, left_raw.shape).astype(np.float32)
        rnoise = np.random.normal(0, 0.008, right_raw.shape).astype(np.float32)
        l2 = left_raw  + (lnoise if np.sum(np.abs(left_raw))  > 0 else 0)
        r2 = right_raw + (rnoise if np.sum(np.abs(right_raw)) > 0 else 0)
        pairs.append((l2, r2))
    return pairs


# ─────────────────────────────────────────────
# DETECT BOTH HANDS WITH FALLBACK STRATEGIES
# ─────────────────────────────────────────────

def detect_hands(frame, hands_ctx):
    """
    Try original, padded, brightness variants.
    Returns best (left_raw, right_raw, meta, overlapping).
    """
    h, w = frame.shape[:2]
    padded = cv2.copyMakeBorder(frame,40,40,40,40,
                cv2.BORDER_CONSTANT, value=[128,128,128])
    bright = cv2.convertScaleAbs(frame, alpha=1.15, beta=15)
    dark   = cv2.convertScaleAbs(frame, alpha=0.85, beta=-10)

    best_left  = np.zeros(SINGLE_HAND_SIZE, dtype=np.float32)
    best_right = np.zeros(SINGLE_HAND_SIZE, dtype=np.float32)
    best_meta  = []
    best_overlap = False
    best_count = 0

    for img in [frame, padded, bright, dark]:
        ih, iw = img.shape[:2]
        _, results = mediapipe_detection(img, hands_ctx)
        if not (results and results.multi_hand_landmarks):
            continue
        found = len(results.multi_hand_landmarks)
        if found > best_count:
            best_count = found
            best_left, best_right, best_meta, best_overlap = \
                assign_hands_robust(results, iw, ih)
        if best_count == 2:
            break

    return best_left, best_right, best_meta, best_overlap


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def collect():
    os.makedirs(DATA_PATH, exist_ok=True)
    total_saved = total_failed = total_partial = 0

    with mp_hands_sol.Hands(
        static_image_mode=True, max_num_hands=2,
        model_complexity=0, min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as hands:

        for action in ACTIONS:
            print(f"\n📂  Class: {action}")
            img_dir = os.path.join(IMAGE_PATH, action)
            files   = sorted(
                glob.glob(os.path.join(img_dir,"*.jpg")) +
                glob.glob(os.path.join(img_dir,"*.png")) +
                glob.glob(os.path.join(img_dir,"*.jpeg"))
            )[:NO_SEQUENCES]

            if not files:
                print(f"  ⚠️  No images — skipping"); continue

            saved = failed = partial = 0

            for fpath in files:
                frame = cv2.imread(fpath)
                if frame is None: continue

                enh = enhance(frame)

                # Image-space augmentations
                for aug_img in augment_images(enh):
                    l_raw, r_raw, _, _ = detect_hands(aug_img, hands)

                    lf = np.sum(np.abs(l_raw)) != 0
                    rf = np.sum(np.abs(r_raw)) != 0

                    if not lf and not rf:
                        failed += 1
                        continue
                    if lf != rf:
                        partial += 1

                    # Keypoint-space augmentations on top
                    for l2, r2 in augment_keypoints(l_raw, r_raw, n=1):
                        vec = build_feature_vector(l2, r2)

                        assert vec.shape[0] == EFFECTIVE_FEATURE, \
                            f"Shape mismatch: {vec.shape[0]} != {EFFECTIVE_FEATURE}"

                        sdir = os.path.join(DATA_PATH, action, str(saved))
                        os.makedirs(sdir, exist_ok=True)
                        np.save(os.path.join(sdir, "0.npy"), vec)
                        saved += 1

            total_saved   += saved
            total_failed  += failed
            total_partial += partial
            rate = 100 * saved / max(len(files)*12*3, 1)
            print(f"  ✅  {saved} saved | {partial} partial | {failed} failed")

    print(f"\n{'─'*50}")
    print(f"  Total saved   : {total_saved}")
    print(f"  Total partial : {total_partial}")
    print(f"  Total failed  : {total_failed}")
    print(f"  Feature size  : {EFFECTIVE_FEATURE}")
    print(f"{'─'*50}\n")

    print("📊  Samples per class:")
    for action in ACTIONS:
        d = os.path.join(DATA_PATH, action)
        c = len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d,x))]) \
            if os.path.exists(d) else 0
        print(f"  {action}: {c:5d}  {'✅' if c >= 500 else '⚠️ LOW'}")
    print("\n🎉  Done.")


if __name__ == "__main__":
    collect()