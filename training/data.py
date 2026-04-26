"""
data.py
-------
Optional live webcam image capture for building / extending your dataset.

Since you already have an image dataset, this file is kept but
is NOT required for inference.

Controls
--------
  Press the letter key (a–z) to save the current ROI as
  Images/<LETTER>/<count>.jpg

  Press 'Q' to quit.
"""

import os
import cv2

from config import IMAGE_PATH, ACTIONS


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

cap = cv2.VideoCapture(0)
os.makedirs(IMAGE_PATH, exist_ok=True)

print("📷  Webcam data collector started.")
print("    Press a letter key to save that gesture's image.")
print("    Press 'Q' to quit.\n")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ROI for the hand
    roi = frame[40:400, 0:300]

    # Visualise ROI box
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.putText(
        frame,
        "Place hand in box | Press letter to save | Q = quit",
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (200, 200, 200), 1, cv2.LINE_AA,
    )

    cv2.imshow("Data Collector", frame)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        break

    # Map key to uppercase letter and save if it is a valid action
    char = chr(key).upper()
    if char in ACTIONS:
        save_path = os.path.join(IMAGE_PATH, char)
        os.makedirs(save_path, exist_ok=True)
        count = len([f for f in os.listdir(save_path) if f.endswith(".jpg")])
        filename = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(filename, roi)
        print(f"  ✅  Saved {char} image #{count} → {filename}")


cap.release()
cv2.destroyAllWindows()
print("👋  Data collection stopped.")
