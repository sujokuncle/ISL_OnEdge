# 🧠 Training Pipeline

This folder contains all scripts required to:

* Extract features from image datasets
* Train the model
* Perform augmentation
* Evaluate and diagnose model performance

---

## 📂 Files Overview

| File                   | Purpose                                                                   | Key Decision                                                                |
| ---------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `collectdata.py`       | Extracts hand landmarks from images and saves 196D `.npy` feature vectors | Detect once, augment keypoints → much faster than repeated image processing |
| `trainmodel.py`        | Trains the Residual MLP model and exports it to TFLite                    | Uses class balancing + early stopping for stable training                   |
| `overlapaugment.py`    | Generates synthetic overlapping hand samples                              | Blends xyz coordinates without affecting geometric features                 |
| `convert_to_tflite.py` | Converts trained model to TFLite format                                   | Enables edge deployment (Raspberry Pi)                                      |
| `data.py`              | Optional script for collecting live dataset using webcam                  | Useful for custom dataset creation                                          |

---

## 📊 Evaluation & Diagnosis

| File                | Purpose                                                              |
| ------------------- | -------------------------------------------------------------------- |
| `check_accuracy.py` | Evaluates model on dataset and generates accuracy + confusion matrix |
| `diagnosis2.py`     | Separates threshold issues from actual model confusion               |
| `live_diagnosis.py` | Visualizes real-time prediction confidence                           |

---

## ⚙️ Pipeline Workflow

### Step 1 — Extract Keypoints

```bash
python collectdata.py
```

* Reads images from `Images/`
* Saves processed features into `MP_Data/`
* Output: 196D `.npy` files

---

### Step 2 — Data Augmentation (Optional)

```bash
python overlapaugment.py
```

* Simulates overlapping hand gestures
* Improves robustness

---

### Step 3 — Train Model

```bash
python trainmodel.py
```

* Loads `.npy` dataset
* Trains Residual MLP
* Exports `model.tflite`

---

### Step 4 — Convert to TFLite (if needed separately)

```bash
python convert_to_tflite.py
```

---

### Step 5 — Evaluate Model

```bash
python check_accuracy.py
```

Outputs:

* `accuracy_report.png`
* `confusion_matrix.png`

---

### Step 6 — Diagnose Errors

```bash
python diagnosis2.py
```

* Identifies whether errors are due to:

  * Threshold tuning
  * Data quality

---

## 📌 Dataset Requirement

This pipeline expects:

```bash
Images/
  A/
  B/
  ...
  Z/

MP_Data/
  A/
  B/
  ...
  Z/
```

⚠️ These folders are NOT included in the repository (see `datasets/README.md`)

---

## 💡 Key Insight

Instead of training on raw images, the system uses **geometric features**:

* 63 → normalized coordinates
* 20 → bone lengths
* 15 → joint angles

👉 Total: **196D feature vector (2 hands)**

This makes the model:

* Smaller
* Faster
* More robust

---

## 🎯 Goal

The training pipeline is designed to:

* Be efficient (minimal recomputation)
* Be lightweight (edge deployment ready)
* Be extensible (future word-level recognition)
