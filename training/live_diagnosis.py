# live_diagnosis.py
# Shows raw predictions from webcam in real time
# so you can see exactly what the model thinks for each gesture
# Run: python live_diagnosis.py

import cv2
import sys
import numpy as np
import mediapipe as mp

from config     import EFFECTIVE_FEATURE, ACTIONS
from detection  import mediapipe_detection, assign_hands_robust, \
                       draw_styled_landmarks, build_hands_model
from preprocessing import build_feature_vector
from inference  import GestureClassifier

classifier = GestureClassifier()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands_mod = mp.solutions.hands

with build_hands_model(static_image_mode=False) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        image, results = mediapipe_detection(frame, hands)
        draw_styled_landmarks(image, results)

        left_raw, right_raw, hands_meta, overlapping = \
            assign_hands_robust(results, w, h)

        features = build_feature_vector(left_raw, right_raw)

        left_detected  = np.sum(np.abs(left_raw))  != 0
        right_detected = np.sum(np.abs(right_raw)) != 0

        # Raw prediction — no threshold applied
        _, _, probs = classifier.predict(features)
        top5_idx    = np.argsort(probs)[::-1][:5]

        # Background panel
        cv2.rectangle(image, (0,0), (w,55), (30,30,30), -1)

        # Hand detection status
        l_color = (0,255,0) if left_detected  else (0,0,255)
        r_color = (0,255,0) if right_detected else (0,0,255)
        cv2.putText(image, f"L:{left_detected}",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, l_color, 2)
        cv2.putText(image, f"R:{right_detected}",
                    (130,35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, r_color, 2)

        if overlapping:
            cv2.putText(image, "OVERLAP",
                        (250,35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,165,255), 2)

        # Top 5 predictions with confidence bars
        for rank, idx in enumerate(top5_idx):
            label = ACTIONS[idx]
            prob  = probs[idx]
            bar_w = int(prob * 200)
            y0    = 65  + rank * 40
            y1    = y0  + 28

            # Color: green if top, orange if 2nd, gray rest
            if rank == 0:
                color = (0,200,0)
            elif rank == 1:
                color = (0,140,255)
            else:
                color = (80,80,80)

            cv2.rectangle(image, (0,y0), (bar_w,y1), color, -1)
            cv2.putText(image,
                        f"{label}: {prob*100:.1f}%",
                        (bar_w+5, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255,255,255), 2)

        # Threshold line marker
        thresh_x = int(0.40 * 200)
        cv2.line(image,
                 (thresh_x, 65),
                 (thresh_x, 65 + 5*40),
                 (0,0,255), 2)
        cv2.putText(image, "threshold",
                    (thresh_x+3, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0,0,255), 1)

        cv2.putText(image, "Q=quit",
                    (w-80, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (150,150,150), 1)

        cv2.imshow("Live Diagnosis", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()