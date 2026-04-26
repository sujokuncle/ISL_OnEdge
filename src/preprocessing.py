"""
preprocessing.py
----------------
Rich feature extraction for ISL hand gestures.

Feature vector per hand (98 values):
  • 63  — normalised xyz coordinates (wrist-origin, scale-invariant)
  • 20  — bone lengths (distance between connected joints)
  • 15  — joint angles (angle at each knuckle)

Two-hand combined: 98 × 2 = 196 features

Why richer features?
  Raw xyz is affected by hand orientation.
  Bone lengths + angles are rotation-invariant and capture
  finger geometry much better — especially for overlapping hands
  where position coordinates become unreliable.
"""

import numpy as np
from config import NUM_KEYPOINTS, SINGLE_HAND_SIZE, RICH_SINGLE, USE_RICH_FEATURES

# MediaPipe hand bone connections (parent → child joint index pairs)
BONES = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20) # pinky
]  # 20 bones

# Joint triplets for angle computation (A-B-C → angle at B)
ANGLE_TRIPLETS = [
    (0,1,2),(1,2,3),(2,3,4),       # thumb
    (0,5,6),(5,6,7),(6,7,8),       # index
    (0,9,10),(9,10,11),(10,11,12), # middle
    (0,13,14),(13,14,15),(14,15,16),# ring
    (0,17,18),(17,18,19),(18,19,20) # pinky
]  # 15 angles


def _angle(a, b, c):
    """Angle in degrees at point B in triangle A-B-C."""
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def extract_raw_keypoints(results, hand_index: int = 0) -> np.ndarray:
    """
    Extract raw (un-normalised) xyz for ONE hand.
    Returns np.ndarray of shape (63,) — zeros if not found.
    """
    if results and results.multi_hand_landmarks:
        if hand_index < len(results.multi_hand_landmarks):
            lms = results.multi_hand_landmarks[hand_index].landmark
            return np.array([[lm.x, lm.y, lm.z] for lm in lms],
                            dtype=np.float32).flatten()
    return np.zeros(SINGLE_HAND_SIZE, dtype=np.float32)


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalise a (63,) raw keypoint vector.
    - Translate wrist to origin
    - Scale to [-1, 1]
    Returns (63,) float32.
    """
    if np.sum(np.abs(keypoints)) == 0:
        return np.zeros(SINGLE_HAND_SIZE, dtype=np.float32)

    kp = keypoints[:SINGLE_HAND_SIZE].reshape(NUM_KEYPOINTS, 3).copy()
    kp -= kp[0]                          # wrist → origin
    mx = np.max(np.abs(kp))
    if mx > 0:
        kp /= mx
    return kp.flatten().astype(np.float32)


def compute_bone_lengths(kp_xyz: np.ndarray) -> np.ndarray:
    """
    Compute 20 bone lengths from a (21,3) landmark array.
    Normalised by the wrist-to-middle-finger-mcp distance
    so the vector is scale-invariant.
    Returns (20,) float32.
    """
    lengths = np.array(
        [np.linalg.norm(kp_xyz[b] - kp_xyz[a]) for a, b in BONES],
        dtype=np.float32
    )
    ref = np.linalg.norm(kp_xyz[9] - kp_xyz[0]) + 1e-8  # wrist→middle MCP
    return lengths / ref


def compute_joint_angles(kp_xyz: np.ndarray) -> np.ndarray:
    """
    Compute 15 joint angles (degrees, normalised to [0,1]).
    Returns (15,) float32.
    """
    angles = np.array(
        [_angle(kp_xyz[a], kp_xyz[b], kp_xyz[c])
         for a, b, c in ANGLE_TRIPLETS],
        dtype=np.float32
    )
    return angles / 180.0   # normalise to [0,1]


def rich_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Build a 98-feature vector for ONE hand:
      [normalised_xyz(63) | bone_lengths(20) | joint_angles(15)]

    Parameters
    ----------
    keypoints : raw (63,) array from extract_raw_keypoints()

    Returns
    -------
    np.ndarray of shape (98,) — zeros if hand not detected
    """
    if np.sum(np.abs(keypoints)) == 0:
        return np.zeros(RICH_SINGLE, dtype=np.float32)

    kp_xyz = keypoints[:SINGLE_HAND_SIZE].reshape(NUM_KEYPOINTS, 3).copy()

    norm_xyz = normalize_keypoints(keypoints)        # (63,)
    bones    = compute_bone_lengths(kp_xyz)          # (20,)
    angles   = compute_joint_angles(kp_xyz)          # (15,)

    return np.concatenate([norm_xyz, bones, angles]).astype(np.float32)


def build_feature_vector(left_raw: np.ndarray,
                         right_raw: np.ndarray) -> np.ndarray:
    """
    Build the final feature vector from two raw (63,) hand arrays.

    If USE_RICH_FEATURES → returns (196,) = rich_left(98) + rich_right(98)
    Else                  → returns (126,) = norm_left(63) + norm_right(63)
    """
    if USE_RICH_FEATURES:
        left_feat  = rich_features(left_raw)
        right_feat = rich_features(right_raw)
    else:
        left_feat  = normalize_keypoints(left_raw)
        right_feat = normalize_keypoints(right_raw)

    return np.concatenate([left_feat, right_feat]).astype(np.float32)


def get_handedness(results, hand_index: int = 0) -> str:
    if results and results.multi_handedness:
        if hand_index < len(results.multi_handedness):
            return results.multi_handedness[hand_index].classification[0].label
    return "Unknown"