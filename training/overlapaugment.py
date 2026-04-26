"""
overlap_augment.py
------------------
Run this AFTER collectdata.py has finished.
It reads existing clean MP_Data samples and generates
synthetic overlapping versions by blending hand positions.

This teaches the model what overlapping hands look like
without needing to re-collect images.

Usage:
    python overlap_augment.py
"""

import os
import numpy as np
from config import DATA_PATH, ACTIONS, EFFECTIVE_FEATURE, SINGLE_HAND_SIZE

RICH_SINGLE = EFFECTIVE_FEATURE // 2   # 98 features per hand


def blend_hands(left_feat, right_feat, blend_factor):
    """
    Simulate overlapping hands by blending xyz coordinates
    of left and right hands toward each other.

    blend_factor 0.0 = no overlap (original)
    blend_factor 0.5 = heavy overlap (hands merged)
    blend_factor 1.0 = complete overlap (same position)

    Only blends xyz part (first 63 values of each hand).
    Bone lengths and angles stay the same — they are
    shape features and don't change with position.
    """
    left_out  = left_feat.copy()
    right_out = right_feat.copy()

    # xyz is first 63 values of each 98-feature hand vector
    XYZ = SINGLE_HAND_SIZE  # 63

    left_xyz  = left_feat[:XYZ].copy()
    right_xyz = right_feat[:XYZ].copy()

    # Blend xyz toward each other
    left_out[:XYZ]  = left_xyz  * (1 - blend_factor) + \
                      right_xyz * blend_factor
    right_out[:XYZ] = right_xyz * (1 - blend_factor) + \
                      left_xyz  * blend_factor

    return left_out, right_out


def generate_overlap_samples():
    total_added = 0

    for action in ACTIONS:
        action_dir = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_dir):
            print(f'  ⚠️  Missing: {action}')
            continue

        # Get all existing sample folders
        folders = sorted(
            [d for d in os.listdir(action_dir)
             if os.path.isdir(os.path.join(action_dir, d))],
            key=lambda x: int(x)
        )

        if not folders:
            continue

        # Find the next available index to save new samples
        next_idx = max([int(f) for f in folders]) + 1
        added    = 0

        for folder in folders:
            npy_path = os.path.join(action_dir, folder, '0.npy')
            if not os.path.exists(npy_path):
                continue

            vec = np.load(npy_path)
            if vec.shape[0] != EFFECTIVE_FEATURE:
                continue

            # Split into left and right hand features
            left_feat  = vec[:RICH_SINGLE].copy()
            right_feat = vec[RICH_SINGLE:].copy()

            left_has  = np.sum(np.abs(left_feat))  != 0
            right_has = np.sum(np.abs(right_feat)) != 0

            # Only augment samples where both hands are present
            if not left_has or not right_has:
                continue

            # Generate overlap variants at different blend levels
            for blend in [0.2, 0.35, 0.50]:
                l_blend, r_blend = blend_hands(
                    left_feat, right_feat, blend
                )
                overlap_vec = np.concatenate(
                    [l_blend, r_blend]
                ).astype(np.float32)

                save_dir = os.path.join(
                    DATA_PATH, action, str(next_idx)
                )
                os.makedirs(save_dir, exist_ok=True)
                np.save(os.path.join(save_dir, '0. npy'), overlap_vec)
                next_idx += 1
                added    += 1

        total_added += added
        print(f'  {action}: +{added} overlap samples added')

    print(f'\nTotal overlap samples added: {total_added}')
    print('\nSamples per class after augmentation:')
    for action in ACTIONS:
        d = os.path.join(DATA_PATH, action)
        c = len([x for x in os.listdir(d)
                 if os.path.isdir(os.path.join(d, x))]) \
            if os.path.exists(d) else 0
        flag = '✅' if c >= 500 else '⚠️  LOW'
        print(f'  {action}: {c:6d}  {flag}')


if __name__ == '__main__':
    print('Generating overlap augmentation samples...\n')
    generate_overlap_samples()
    print('\n✅ Done. Now re-zip MP_Data and retrain on Colab.')