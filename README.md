# SPC4004 Assessment 2 — Skin Cancer Classification

A deep learning project for multi-class classification of dermoscopic skin lesion images using the HAM10000 dataset. The project follows an iterative experimental methodology: starting from a simple baseline CNN, systematically tuning hyperparameters across 5 runs, and then training a final optimised model.

---

## Table of Contents

- [Project Purpose](#project-purpose)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
  - [first\_skin\_cancer\_classification.py](#1-first_skin_cancer_classificationpy--baseline-cnn)
  - [skin\_cancer\_iterative.py](#2-skin_cancer_iterativepy--5-run-hyperparameter-study)
  - [skin\_cancer\_final.py](#3-skin_cancer_finalpy--final-optimised-model-run-6)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [Running the Scripts](#running-the-scripts)

---

## Project Purpose

This project investigates how incrementally applying deep learning best practices affects skin lesion classification accuracy. The seven diagnostic classes span both malignant and benign conditions, and the dataset has severe class imbalance (the dominant class `nv` has ~58× more samples than the rarest class `df`). The experiments explore:

- Baseline CNN performance
- The effect of BatchNormalization, data augmentation, and class weight balancing
- Optimizer choice (Adam vs. SGD with Nesterov momentum)
- A final combined approach using a deeper architecture, L2 regularisation, and a `tf.data` pipeline

---

## Dataset

**HAM10000** ("Human Against Machine with 10000 training images")

| Class | Description | Type |
|-------|-------------|------|
| `akiec` | Actinic keratoses & intraepithelial carcinoma | Malignant |
| `bcc` | Basal cell carcinoma | Malignant |
| `bkl` | Benign keratosis-like lesions | Benign |
| `df` | Dermatofibroma | Benign |
| `mel` | Melanoma | Malignant |
| `nv` | Melanocytic nevi | Benign (dominant ~6,700 samples) |
| `vasc` | Vascular lesions | Benign |

The dataset is **not included** in this repository due to its size (~3 GB). Images must be placed in the following directories before running any script:

```
HAM10000_images_part_1/
HAM10000_images_part_2/
HAM10000_metadata.csv
```

Download from: https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection

---

## Project Structure

```
SPC4004-assessment-2/
├── requirements/
│   ├── requirements.txt        # pip dependencies
│   └── pyproject.toml          # project metadata
├── scripts/
│   ├── first_skin_cancer_classification.py   # Baseline CNN
│   ├── skin_cancer_iterative.py              # 5-run hyperparameter study
│   └── skin_cancer_final.py                  # Final optimised model (Run 6)
└── results/
    ├── keras_weights/          # Saved model weights (.keras)
    ├── confusion_matrix/       # Confusion matrix PNGs
    ├── training_curve/         # Accuracy & loss curve PNGs
    └── comparison/             # Cross-run comparison chart and CSV
```

---

## Scripts

### 1. `first_skin_cancer_classification.py` — Baseline CNN

A simple 3-block CNN with no augmentation, no BatchNormalization, and no class weight balancing. Serves as the control experiment.

**Architecture:**
```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
→ Flatten → Dense(128, relu) → Dropout(0.5) → Dense(7, softmax)
```

**Configuration:**
| Parameter | Value |
|-----------|-------|
| Image size | 128×128 |
| Batch size | 32 |
| Epochs | 25 (EarlyStopping patience=5) |
| Optimizer | Adam (lr=1e-3) |
| Data split | 70% train / 15% val / 15% test (stratified) |

**Outputs:**
- `results/confusion_matrix/baseline_cm.png`
- `results/training_curve/baseline_training_curve.png`

---

### 2. `skin_cancer_iterative.py` — 5-Run Hyperparameter Study

Runs 5 sequential experiments, each building on or pivoting from the previous, to identify which techniques improve generalisation.

| Run | Key Changes | Test Acc |
|-----|-------------|----------|
| 1 | Exact baseline replication | 0.7512 |
| 2 | + BatchNormalization, lr → 5e-4 | 0.7073 |
| 3 | + Data augmentation (flip, rotate, zoom) | 0.7452 |
| 4 | + Balanced class weights | 0.6367 |
| 5 | SGD + Nesterov momentum, ReduceLROnPlateau | 0.5569 |

**How it works:**

- Each run is defined by a configuration dictionary specifying optimizer, learning rate, batch size, max epochs, EarlyStopping patience, and flags for augmentation/BatchNorm/class weights.
- After all 5 runs, metrics are written to `results/comparison/comparison_log.csv` and a bar chart is saved to `results/comparison/runs_comparison.png`.
- Automatically detects GPU and enables mixed-precision (float16) training when available.

**Per-run outputs:**
- `results/keras_weights/run{N}_best.keras`
- `results/confusion_matrix/run{N}_cm.png`
- `results/training_curve/run{N}_training_curve.png`

---

### 3. `skin_cancer_final.py` — Final Optimised Model (Run 6)

Incorporates lessons from Runs 1–5 into a single, more capable model.

**Key improvements over earlier runs:**

| Technique | Detail |
|-----------|--------|
| Deeper architecture | 5 Conv blocks: 32 → 64 → 128 → 256 → 512 filters |
| GlobalAveragePooling2D | Replaces Flatten; better suited to deep nets |
| BatchNormalization | After every Conv2D layer |
| L2 regularisation | weight_decay=1e-4 on all Conv2D and Dense kernels |
| Augmentation | Flip, rotation (0.15), zoom (0.1), contrast (0.15) |
| Class weights | Balanced weighting via `compute_class_weight` |
| Optimizer | Adam (lr=3e-4) |
| ReduceLROnPlateau | Halves LR when val_loss plateaus (patience=4) |
| ModelCheckpoint | Saves only the best-performing epoch |
| EarlyStopping | patience=8, restores best weights |
| tf.data pipeline | Shuffle, batch, and prefetch for efficient GPU feeding |
| Mixed precision | float16 on GPU for faster training |

**Architecture:**
```
[Conv2D(32) + BN + MaxPool] × 1
[Conv2D(64) + BN + MaxPool] × 1
[Conv2D(128) + BN + MaxPool] × 1
[Conv2D(256) + BN + MaxPool] × 1
[Conv2D(512) + BN + MaxPool] × 1
→ GlobalAveragePooling2D
→ Dense(512, relu) → Dropout(0.5) → Dense(7, softmax)
```

**Outputs:**
- `results/keras_weights/final_run6_best.keras`
- `results/confusion_matrix/final_run_cm.png`
- `results/training_curve/final_performance.png`
- `results/training_curve/final_lr_schedule.png`

---

## Results

All result files are tracked in git. Key comparison artifacts:

| File | Description |
|------|-------------|
| `results/comparison/comparison_log.csv` | Per-run metrics table |
| `results/comparison/runs_comparison.png` | Bar chart: test accuracy & val loss across runs |
| `results/confusion_matrix/` | One confusion matrix PNG per run |
| `results/training_curve/` | Accuracy and loss curves per run |

---

## Setup & Installation

**Requirements:** Python 3.13+, pip

### 1. Clone the repository

```bash
git clone <repo-url>
cd SPC4004-assessment-2
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements/requirements.txt
```

Or, if using the pyproject.toml approach:

```bash
pip install -e requirements/
```

### 4. Download the dataset

Download the HAM10000 dataset from Kaggle and place the files so the directory layout looks like:

```
SPC4004-assessment-2/
├── HAM10000_images_part_1/   ← ~5,000 JPEG images
├── HAM10000_images_part_2/   ← ~5,000 JPEG images
├── HAM10000_metadata.csv
└── scripts/
    └── ...
```

### 5. (Optional) GPU acceleration

Install the appropriate CUDA toolkit and cuDNN for your GPU. TensorFlow will automatically detect and use the GPU. Mixed-precision training is enabled automatically when a compatible GPU is found.

---

## Running the Scripts

All scripts are run from the repository root. They automatically locate the dataset and save results to the `results/` directory.

### Baseline

```bash
python scripts/first_skin_cancer_classification.py
```

### 5-Run Iterative Study

```bash
python scripts/skin_cancer_iterative.py
```

This script trains 5 models sequentially. Expect the full run to take 2–3 hours on CPU, or 30–60 minutes on a modern GPU.

### Final Model (Run 6)

```bash
python scripts/skin_cancer_final.py
```

Results are saved under `results/`. The best model weights are saved to `results/keras_weights/final_run6_best.keras`.
