"""
check_accuracy.py
-----------------
Tests the trained model on your entire MP_Data folder
and gives a complete accuracy report.

Run: python check_accuracy.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config     import DATA_PATH, ACTIONS, EFFECTIVE_FEATURE
from inference  import GestureClassifier

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
SAMPLES_PER_CLASS = 200   # how many samples to test per class
                           # increase to 500 for more accurate result
                           # but takes longer

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print('Loading model...')
classifier = GestureClassifier()
print('Model loaded ✅\n')

# ─────────────────────────────────────────────
# TEST EACH CLASS
# ─────────────────────────────────────────────
all_true  = []
all_pred  = []
all_conf  = []

class_stats = {}

print(f'Testing {SAMPLES_PER_CLASS} samples per class...\n')
print(f"{'Letter':<8} {'Tested':<8} {'Correct':<9} "
      f"{'Accuracy':<10} {'AvgConf':<9} {'Status'}")
print('─' * 60)

t_start = time.time()

for action in ACTIONS:
    action_dir = os.path.join(DATA_PATH, action)

    if not os.path.exists(action_dir):
        print(f'  {action:<6}  ⚠️  folder missing')
        continue

    folders = sorted(
        [d for d in os.listdir(action_dir)
         if os.path.isdir(os.path.join(action_dir, d))],
        key=lambda x: int(x)
    )

    # Randomly sample for fairness
    np.random.shuffle(folders)
    folders = folders[:SAMPLES_PER_CLASS]

    correct     = 0
    total       = 0
    confidences = []
    wrong_preds = defaultdict(int)

    for folder in folders:
        npy = os.path.join(action_dir, folder, '0.npy')
        if not os.path.exists(npy):
            continue

        kp = np.load(npy)
        if kp.shape[0] != EFFECTIVE_FEATURE:
            continue

        pred_label, confidence, _ = classifier.predict(kp)
        total      += 1
        confidences.append(confidence)

        all_true.append(action)
        all_pred.append(pred_label if pred_label != '?' else 'UNK')
        all_conf.append(confidence)

        if pred_label == action:
            correct += 1
        else:
            wrong_preds[pred_label] += 1

    if total == 0:
        continue

    accuracy = 100 * correct / total
    avg_conf = np.mean(confidences) * 100

    if accuracy >= 90:
        status = '✅ Excellent'
    elif accuracy >= 75:
        status = '🟡 Good'
    elif accuracy >= 50:
        status = '🟠 Fair'
    else:
        status = '❌ Poor'

    top_wrong = sorted(wrong_preds.items(),
                       key=lambda x: x[1],
                       reverse=True)[:2]
    confused  = ' '.join([f"{l}({n})" for l,n in top_wrong]) \
                if top_wrong else ''

    class_stats[action] = {
        'accuracy'  : accuracy,
        'avg_conf'  : avg_conf,
        'correct'   : correct,
        'total'     : total,
        'confused'  : confused,
    }

    print(f'{action:<8} {total:<8} {correct:<9} '
          f'{accuracy:6.1f}%    {avg_conf:6.1f}%   '
          f'{status}  {confused}')

# ─────────────────────────────────────────────
# OVERALL STATS
# ─────────────────────────────────────────────
total_time = time.time() - t_start

total_correct = sum(s['correct'] for s in class_stats.values())
total_tested  = sum(s['total']   for s in class_stats.values())
overall_acc   = 100 * total_correct / max(total_tested, 1)
avg_conf_all  = np.mean(all_conf) * 100 if all_conf else 0

print('\n' + '═' * 60)
print(f'  Overall Accuracy : {overall_acc:.2f}%')
print(f'  Total Tested     : {total_tested}')
print(f'  Total Correct    : {total_correct}')
print(f'  Avg Confidence   : {avg_conf_all:.1f}%')
print(f'  Test Time        : {total_time:.1f}s')
print('═' * 60)

# Grade
if overall_acc >= 95:
    grade = 'A+ — Production ready ✅'
elif overall_acc >= 90:
    grade = 'A  — Excellent ✅'
elif overall_acc >= 80:
    grade = 'B  — Good, minor improvements needed 🟡'
elif overall_acc >= 70:
    grade = 'C  — Fair, needs improvement 🟠'
else:
    grade = 'D  — Poor, retrain recommended ❌'

print(f'\n  Grade: {grade}')

# ─────────────────────────────────────────────
# PROBLEM SUMMARY
# ─────────────────────────────────────────────
poor    = [(a,s) for a,s in class_stats.items() if s['accuracy'] < 70]
fair    = [(a,s) for a,s in class_stats.items()
           if 70 <= s['accuracy'] < 85]
low_conf = [(a,s) for a,s in class_stats.items()
            if s['avg_conf'] < 50 and s['accuracy'] >= 80]

if poor:
    print(f'\n❌ Poor letters (< 70%) — need attention:')
    for a, s in sorted(poor, key=lambda x: x[1]['accuracy']):
        print(f'   {a}: {s["accuracy"]:.1f}%  confused with: {s["confused"]}')

if fair:
    print(f'\n🟠 Fair letters (70–85%) — could improve:')
    for a, s in sorted(fair, key=lambda x: x[1]['accuracy']):
        print(f'   {a}: {s["accuracy"]:.1f}%  confused with: {s["confused"]}')

if low_conf:
    print(f'\n⚡ Low confidence letters — lower threshold helps:')
    for a, s in sorted(low_conf, key=lambda x: x[1]['avg_conf']):
        print(f'   {a}: accuracy={s["accuracy"]:.1f}%  '
              f'conf={s["avg_conf"]:.1f}%')

# ─────────────────────────────────────────────
# PLOT 1 — Per-class accuracy bar chart
# ─────────────────────────────────────────────
if class_stats:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    letters   = list(class_stats.keys())
    accuracies = [class_stats[a]['accuracy'] for a in letters]
    confs      = [class_stats[a]['avg_conf'] for a in letters]

    colors = []
    for acc in accuracies:
        if acc >= 90:   colors.append('#2ecc71')   # green
        elif acc >= 75: colors.append('#f39c12')   # orange
        elif acc >= 50: colors.append('#e67e22')   # dark orange
        else:           colors.append('#e74c3c')   # red

    # Accuracy bars
    bars = axes[0].bar(letters, accuracies, color=colors,
                       edgecolor='white', linewidth=0.5)
    axes[0].axhline(y=90, color='green',  linestyle='--',
                    alpha=0.5, label='90% target')
    axes[0].axhline(y=75, color='orange', linestyle='--',
                    alpha=0.5, label='75% acceptable')
    axes[0].set_title(f'Per-Class Accuracy  |  Overall: {overall_acc:.1f}%',
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(0, 110)
    axes[0].legend(fontsize=9)

    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1,
                     f'{acc:.0f}%',
                     ha='center', va='bottom',
                     fontsize=7, fontweight='bold')

    # Confidence bars
    conf_colors = ['#3498db' if c >= 60 else '#e74c3c' for c in confs]
    axes[1].bar(letters, confs, color=conf_colors,
                edgecolor='white', linewidth=0.5)
    axes[1].axhline(y=60, color='red', linestyle='--',
                    alpha=0.5, label='60% confidence threshold')
    axes[1].set_title('Average Prediction Confidence per Class',
                      fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Avg Confidence (%)')
    axes[1].set_ylim(0, 110)
    axes[1].legend(fontsize=9)

    for bar, conf in zip(axes[1].patches, confs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 1,
                     f'{conf:.0f}%',
                     ha='center', va='bottom',
                     fontsize=7)

    plt.tight_layout()
    plt.savefig('accuracy_report.png', dpi=150, bbox_inches='tight')
    print('\n📊 Accuracy chart saved → accuracy_report.png')

# ─────────────────────────────────────────────
# PLOT 2 — Confusion matrix
# ─────────────────────────────────────────────
if len(all_true) > 0:
    # Replace UNK with ? for display
    pred_labels = [p if p != 'UNK' else '?' for p in all_pred]

    # Only show letters that appear in true labels
    present = sorted(set(all_true))

    cm = confusion_matrix(all_true, pred_labels,
                          labels=present + (['?'] if '?' in pred_labels
                                            else []))

    fig2, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(cm,
                annot=True, fmt='d',
                xticklabels=present + (['?'] if '?' in pred_labels else []),
                yticklabels=present,
                cmap='Blues',
                ax=ax,
                linewidths=0.3)

    ax.set_title(f'Confusion Matrix  |  Overall Accuracy: {overall_acc:.1f}%',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label',      fontsize=12)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print('📊 Confusion matrix saved → confusion_matrix.png')
    plt.show()

print('\n✅ Accuracy check complete.')
print('Check accuracy_report.png and confusion_matrix.png for visual reports.')