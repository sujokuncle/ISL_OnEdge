"""
diagnosis2.py
-------------
Deep diagnosis — shows:
1. What happens when threshold is removed completely
2. Top 3 predictions per letter regardless of confidence
3. Average confidence per letter
4. Identifies if problem is threshold or genuine confusion

Run: python diagnosis2.py
"""

import numpy as np
import os
from config import DATA_PATH, ACTIONS, EFFECTIVE_FEATURE
from inference import GestureClassifier

classifier = GestureClassifier()

print("Deep diagnosis — threshold removed, showing raw predictions\n")
print(f"{'Letter':<6} {'Acc%':<7} {'AvgConf':<9} "
      f"{'Top prediction':<16} {'2nd':<14} {'3rd':<14} {'Diagnosis'}")
print("─" * 95)

problem_threshold = []   # fixed by lowering threshold
problem_data      = []   # needs more/better data
problem_confused  = []   # confused with similar letters

for action in ACTIONS:
    action_dir = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_dir):
        continue

    folders = sorted(
        [d for d in os.listdir(action_dir)
         if os.path.isdir(os.path.join(action_dir, d))],
        key=lambda x: int(x)
    )

    correct      = 0
    total        = 0
    confidences  = []
    top_preds    = {}

    for folder in folders[:100]:
        npy = os.path.join(action_dir, folder, '0.npy')
        if not os.path.exists(npy):
            continue
        kp = np.load(npy)
        if kp.shape[0] != EFFECTIVE_FEATURE:
            continue

        # Get raw probs — ignore threshold completely
        _, _, probs = classifier.predict(kp)

        # Top 3 raw predictions
        top3_idx  = np.argsort(probs)[::-1][:3]
        top3      = [(ACTIONS[i], float(probs[i])) for i in top3_idx]
        top_label = top3[0][0]
        top_conf  = top3[0][1]

        confidences.append(top_conf)
        top_preds[top_label] = top_preds.get(top_label, 0) + 1
        total += 1

        if top_label == action:
            correct += 1

    if total == 0:
        continue

    accuracy    = 100 * correct / total
    avg_conf    = np.mean(confidences) * 100
    top3_labels = sorted(top_preds.items(),
                         key=lambda x: x[1], reverse=True)[:3]

    t1 = f"{top3_labels[0][0]}({top3_labels[0][1]})" \
         if len(top3_labels) > 0 else "-"
    t2 = f"{top3_labels[1][0]}({top3_labels[1][1]})" \
         if len(top3_labels) > 1 else "-"
    t3 = f"{top3_labels[2][0]}({top3_labels[2][1]})" \
         if len(top3_labels) > 2 else "-"

    # Diagnose
    if accuracy >= 80 and avg_conf < 65:
        diagnosis = "⚡ THRESHOLD — lower to 0.40"
        problem_threshold.append(action)
    elif accuracy >= 80:
        diagnosis = "✅ OK"
    elif accuracy < 50 and top3_labels[0][0] != action:
        diagnosis = f"🔀 CONFUSED with {top3_labels[0][0]}"
        problem_confused.append(action)
    else:
        diagnosis = "📦 DATA — needs more samples"
        problem_data.append(action)

    flag = "✅" if accuracy >= 80 else "❌"
    print(f"{flag} {action:<5} {accuracy:5.1f}%  "
          f"{avg_conf:6.1f}%   "
          f"{t1:<16} {t2:<14} {t3:<14} {diagnosis}")

print("\n" + "─" * 95)
print(f"\n⚡ Fix by lowering threshold : {problem_threshold}")
print(f"🔀 Fix by more diverse data  : {problem_confused}")
print(f"📦 Fix by more/better data   : {problem_data}")

print("\n\nRECOMMENDED ACTIONS:")
if problem_threshold:
    print(f"\n1. In config.py set CONFIDENCE_THRESHOLD = 0.40")
    print(f"   This will fix: {problem_threshold}")

if problem_confused:
    print(f"\n2. These letters are confused with similar ones:")
    for l in problem_confused:
        print(f"   {l} — add more diverse training images")

if problem_data:
    print(f"\n3. These letters need better training data:")
    for l in problem_data:
        print(f"   {l} — check Images/{l}/ folder quality")
    print("   Run: python check_images.py  (to inspect these classes)")