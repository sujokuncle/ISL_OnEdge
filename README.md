# ISL Alphabet Recognition System

> Real-time Indian Sign Language alphabet detection on edge hardware using hand landmark geometry — no image classification, no cloud, no internet.

---

## Overview

This system recognises all 26 letters of the Indian Sign Language alphabet in real time using a standard webcam or Raspberry Pi camera. Instead of training on raw images, it extracts geometric features from hand landmarks detected by MediaPipe — making the model lightweight, fast, and invariant to lighting and background conditions.

The entire inference pipeline runs on a Raspberry Pi 4 at 25–30 FPS with audio output, making it suitable for assistive communication in resource-constrained environments.

---

## Objective

To build a **practical, deployable** sign language recognition system that:

- Works on affordable edge hardware (Raspberry Pi 4, ~$55)
- Runs fully offline — no cloud API, no internet dependency
- Produces both **visual** (on-screen label) and **audio** (text-to-speech) output
- Can be extended to word and sentence recognition in future iterations

---

## System Architecture

```
Camera Frame
     │
     ▼
┌─────────────────────┐
│  CLAHE Enhancement  │  ← closes domain gap between training images and webcam
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  MediaPipe Hands    │  ← detects up to 2 hands, 21 landmarks each (x, y, z)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Feature Extraction │  ← normalised xyz (63) + bone lengths (20) + angles (15)
│  per hand = 98D     │     × 2 hands = 196D feature vector
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TFLite Inference   │  ← Residual MLP, INT8 quantised, ~200 KB
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Smoothing Pipeline │  ← confidence gate → streak lock → majority vote
└────────┬────────────┘
         │
         ▼
   Visual + Audio Output
```

**Why landmarks instead of images?**
Raw images require large datasets, heavy models, and are sensitive to lighting and background. Landmark-based features describe the geometric *shape* of a gesture — they work consistently across different people, skin tones, and environments.

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Hand Detection | MediaPipe Hands | Accurate 21-point 3D landmarks, runs on CPU |
| Feature Engineering | NumPy | Bone lengths + joint angles, rotation/scale invariant |
| Model Training | TensorFlow / Keras | Residual MLP, fast to train on GPU |
| Edge Inference | TensorFlow Lite (INT8) | ~8ms per frame on Pi 4, ~200 KB model size |
| Video Capture | OpenCV | Cross-platform camera access and display |
| Audio Output | espeak / pyttsx3 | Offline TTS — no internet required |
| Training Platform | Google Colab (T4 GPU) | Full pipeline in ~10 minutes vs 2.5 hours on CPU |

---

## Folder Structure

```
ISL-Recognition/
│
├── app.py                  ← main real-time loop
├── config.py               ← all settings in one place
├── detection.py            ← MediaPipe detection + bounding boxes
├── preprocessing.py        ← feature extraction and normalisation
├── inference.py            ← model loading and prediction smoothing
├── utils.py                ← FPS counter, overlays, logging
├── sound_output.py         ← background audio thread
│
├── collectdata.py          ← extract keypoints from image dataset
├── trainmodel.py           ← train and export model
├── data.py                 ← optional live webcam data collection
├── overlap_augment.py      ← augmentation for overlapping hand gestures
├── convert_to_tflite.py    ← standalone TFLite conversion script
│
├── check_accuracy.py       ← full accuracy report with confusion matrix
├── diagnosis2.py           ← per-class threshold vs data quality diagnosis
├── live_diagnosis.py       ← real-time confidence visualiser
│
├── model.tflite            ← INT8 quantised model (~200 KB)
├── requirements.txt
├── .gitignore
└── README.md
```

> `Images/` and `MP_Data/` are not included in the repository due to size.
> See **How to Run** for dataset setup instructions.

---

## Module Explanation

| File | Purpose | Key Decision |
|---|---|---|
| `app.py` | Main pipeline — camera loop, detection, inference, display, audio | Runs inference every 2nd frame to maintain display FPS on Pi |
| `config.py` | Central configuration — paths, actions, thresholds, feature sizes | Single source of truth, no hardcoded values anywhere else |
| `detection.py` | MediaPipe initialisation, bounding box computation, IoU overlap detection | IoU > 0.30 triggers position-based hand assignment instead of trusting MediaPipe labels |
| `preprocessing.py` | Landmark normalisation, bone length and angle feature extraction | Wrist-relative coordinates + scale normalisation make features position and distance invariant |
| `inference.py` | TFLite/Keras model loading, confidence thresholding, streak-lock + majority vote smoothing | Auto-detects TFLite vs Keras — same code runs on Pi and laptop |
| `utils.py` | FPS counter, on-screen probability bars, data logging | Non-critical helpers kept separate to keep app.py readable |
| `sound_output.py` | Background daemon thread with queue for non-blocking TTS | Runs espeak/pyttsx3 in a thread so audio never freezes the video feed |
| `collectdata.py` | Reads image dataset, runs MediaPipe once per image, saves 196D `.npy` keypoint files | Detect once, augment keypoints 6× — 12× faster than re-running detection on augmented images |
| `trainmodel.py` | Loads `.npy` dataset, balances classes, trains Residual MLP, exports TFLite | Class-weighted loss + oversampling handles uneven image counts per letter |
| `overlap_augment.py` | Reads existing MP_Data and generates synthetic overlapping hand samples | Blends xyz coordinates at 4 blend levels without changing bone/angle features |
| `check_accuracy.py` | Tests model on entire MP_Data, outputs per-class accuracy + confusion matrix | Generates `accuracy_report.png` and `confusion_matrix.png` |
| `diagnosis2.py` | Separates threshold failures from genuine confusion | Threshold problem → lower config value. Data problem → collect more images |

---

## Model Details

### Architecture — Residual MLP

```
Input (196)
    → Dense(512) + BatchNorm + Dropout(0.4)
    → ResBlock(512) + ResBlock(256) + ResBlock(128)
    → Dense(64) + Dropout(0.2)
    → Softmax(26)
```

**Why Residual MLP and not LSTM?**
This is a static gesture recognition task — sequence_length = 1. LSTM provides no temporal benefit for single-frame classification. A residual MLP trains faster, generalises better, and exports to a smaller TFLite file.

### Feature Vector — 196 Dimensions

```
Per hand (98):
  ├── Normalised xyz coordinates    63   (wrist at origin, scaled to [-1, 1])
  ├── Bone lengths                  20   (L2 distance between connected joints)
  └── Joint angles                  15   (cosine-based angle at each knuckle)

Two hands:
  [left_hand_98 | right_hand_98]  = 196D total
```

**Why not raw coordinates?**
Raw xyz changes when you move your hand closer to or further from the camera. Bone lengths and joint angles describe the *shape* of the gesture regardless of position, distance, or minor orientation change.

### Training Configuration

| Setting | Value |
|---|---|
| Loss | Categorical Cross-Entropy (label smoothing = 0.1) |
| Optimiser | Adam (lr = 1e-3 with ReduceLROnPlateau) |
| Epochs | Up to 200 with EarlyStopping (patience = 20) |
| Batch size | 128 |
| Class balancing | Oversampling + compute_class_weight('balanced') |
| Training time | ~10 minutes on Google Colab T4 GPU |
| Export | TFLite INT8 quantised (~200 KB) |

---

## Results and Performance

| Metric | Value |
|---|---|
| Overall accuracy | ~88–92% on MP_Data test set |
| Excellent (≥ 90%) | A B E F J K L O P Q S T U V Z |
| Good (75–90%) | C D G I M R W X Y |
| Needs improvement | H N (confused with M) |
| FPS on Raspberry Pi 4 | 25–30 fps @ 320×240 |
| FPS on laptop (CPU) | 40–60 fps @ 640×480 |
| TFLite inference time | ~8ms per frame |
| Model file size | ~200 KB |

**Prediction smoothing reduces visible errors by ~40%** compared to raw per-frame output, by filtering out low-confidence frames and requiring gesture consistency before displaying a result.

---

## Limitations

- **H and N accuracy is lower** (~55–70%) because their hand shapes are geometrically similar to M. More diverse training data from multiple individuals would help.
- **Domain gap** — model was trained on grayscale images with black backgrounds. CLAHE normalisation partially closes this gap for colour webcam input but some accuracy loss remains in very different lighting conditions.
- **Dynamic gestures** — letters J and Z involve hand motion rather than a static pose. The current single-frame approach cannot capture motion. A temporal model (LSTM/Transformer) would be needed.
- **Single-person only** — system is designed for one signer at a time. Multiple signers in frame would confuse hand assignment.
- **No word or sentence context** — each letter is classified independently. The system has no language model to correct unlikely letter sequences.

---

## Target Audience

- **Assistive technology users** — hearing-impaired individuals who communicate via ISL
- **Educators and institutions** — schools and organisations teaching ISL
- **Developers** — as a base system for extending to full ISL word and sentence recognition
- **Recruiters and reviewers** — demonstrating real-world ML engineering on constrained hardware

---

## Why Edge Computing

Running inference on a Raspberry Pi rather than sending video to a cloud API provides three practical advantages for an assistive communication tool:

**1. Privacy** — hand gesture data never leaves the device. No video is transmitted or stored externally.

**2. Latency** — cloud round-trips add 100–500ms of delay. On-device inference takes ~8ms, which is essential for real-time feedback.

**3. Accessibility** — the system works in areas with no internet connectivity, which is important for users in rural or low-connectivity environments where assistive tools are most needed.

---

## Future Scope

- **Word recognition** — extend from alphabet to full ISL word vocabulary using a sequence model over letter predictions
- **Sentence formation** — accumulate words into grammatically structured sentences
- **Bidirectional communication** — text-to-gesture animation so hearing individuals can communicate back
- **Mobile deployment** — port to Android/iOS using TFLite Mobile for smartphone accessibility
- **Multi-person support** — track and assign gestures to individual signers in a shared frame
- **Dynamic gesture support** — add temporal modelling for motion-based letters (J, Z)

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

For Raspberry Pi, install espeak for audio:
```bash
sudo apt install espeak -y
```

### Step 1 — Prepare image dataset

Organise your images in this structure:
```
Images/
  A/  A1.jpg  A2.jpg  ...
  B/  B1.jpg  B2.jpg  ...
  ...
  Z/  Z1.jpg  ...
```

### Step 2 — Extract keypoints

```bash
python collectdata.py
```

This runs MediaPipe on each image once and saves 196D `.npy` feature files to `MP_Data/`.

### Step 3 — Add overlap augmentation (optional but recommended)

```bash
python overlap_augment.py
```

### Step 4 — Train the model

```bash
python trainmodel.py
```

Or train on Google Colab for ~10× speed:
1. Zip `MP_Data/` and upload to Google Drive
2. Open `ISL_Full_Colab.ipynb` in Colab
3. Select T4 GPU runtime and run all cells
4. Download `model.tflite`

### Step 5 — Check accuracy

```bash
python check_accuracy.py
```

### Step 6 — Run real-time recognition

```bash
python app.py
```

Press `Q` to quit. Sound triggers automatically when confidence ≥ 75%.

### Raspberry Pi deployment

Copy these 8 files to the Pi:

```
app.py  config.py  detection.py  preprocessing.py
inference.py  utils.py  sound_output.py  model.tflite
```

```bash
source /home/pi/ISL_Project/isl_env/bin/activate
python app.py
```

---

## Key Insight

> Replacing raw image classification with geometric landmark features (bone lengths + joint angles) reduced the model size from hundreds of megabytes to 200 KB and made it run in real time on a $55 Raspberry Pi — without sacrificing meaningful accuracy.

---

## Contributors

| Name | Role |
|---|---|
| [Your Name] | System design, feature engineering, training, Pi deployment |

---

## Acknowledgements

- [MediaPipe](https://mediapipe.dev) by Google for the hand landmark detection model
- [TensorFlow Lite](https://www.tensorflow.org/lite) for edge inference runtime
- [Google Colab](https://colab.research.google.com) for providing free T4 GPU access for training
- Indian Sign Language dataset contributors
