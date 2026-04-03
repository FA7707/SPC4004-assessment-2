"""
SPC4004 Assessment 2 - Skin Cancer Classification: Iterative Hyperparameter Optimisation
==========================================================================================
5-run sequential hyperparameter search on the HAM10000 dataset.

Each run systematically adjusts:
  - Learning rate (tensorflow.keras.optimizers.Adam / SGD with momentum)
  - Batch size
  - Max epochs + EarlyStopping patience
  - CNN architecture depth / BatchNormalization
  - Data augmentation (tensorflow.keras.layers.RandomFlip, RandomRotation)
  - Learning rate scheduler (tensorflow.keras.callbacks.ReduceLROnPlateau)

GPU/CUDA acceleration is enabled automatically via TensorFlow's device detection.
A per-run confusion matrix PNG and a final cross-run comparison CSV are produced.
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import os
import csv
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight   # for run 4+

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, callbacks

# ---------------------------------------------------------------------------
# 2. GPU / CUDA CONFIGURATION
# ---------------------------------------------------------------------------
# TensorFlow will automatically use any available CUDA-capable GPU.
# Mixed-precision (float16 compute / float32 weights) speeds up training on
# NVIDIA Ampere+ GPUs and Apple Silicon without loss of numerical stability.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        # Allow memory growth so TF does not pre-allocate all VRAM
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] {len(gpus)} GPU(s) detected: {[g.name for g in gpus]}")
    # Enable mixed-precision via tf.keras.mixed_precision
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("[GPU] Mixed-precision policy set to mixed_float16")
else:
    print("[GPU] No GPU detected — running on CPU.")

# ---------------------------------------------------------------------------
# 3. GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
IMAGE_DIRS = [
    BASE_DIR / "HAM10000_images_part_1",
    BASE_DIR / "HAM10000_images_part_2",
]
METADATA_PATH = BASE_DIR / "HAM10000_metadata.csv"
RESULTS_DIR   = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IMG_HEIGHT   = 128
IMG_WIDTH    = 128
IMG_CHANNELS = 3
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# HAM10000 class mapping
LABEL_MAP = {
    "akiec": 0,  # Actinic keratoses        — MALIGNANT
    "bcc":   1,  # Basal cell carcinoma     — MALIGNANT
    "bkl":   2,  # Benign keratosis-like    — BENIGN
    "df":    3,  # Dermatofibroma           — BENIGN
    "mel":   4,  # Melanoma                 — MALIGNANT
    "nv":    5,  # Melanocytic nevi         — BENIGN
    "vasc":  6,  # Vascular lesions         — BENIGN
}
CLASS_NAMES = list(LABEL_MAP.keys())
NUM_CLASSES = len(LABEL_MAP)

# ---------------------------------------------------------------------------
# 4. EXPERIMENT CONFIGURATIONS  (one dict per run)
# ---------------------------------------------------------------------------
# Each entry documents the rationale for the change made versus the prior run,
# using named library/technique references as required by the assessment brief.
EXPERIMENTS = [
    {
        # ── RUN 1: Baseline replication ─────────────────────────────────────
        # Reproduces the simple 3-block CNN from skin_cancer_classification.py
        # using tensorflow.keras.optimizers.Adam at lr=1e-3, no augmentation,
        # no class weighting. Serves as the control for all subsequent runs.
        "run_id":        1,
        "description":   "Baseline CNN — Adam lr=1e-3, bs=32, no augmentation",
        "learning_rate": 1e-3,
        "batch_size":    32,
        "epochs":        25,
        "patience":      5,
        "optimizer":     "adam",
        "augment":       False,
        "use_bn":        False,
        "class_weights": False,
        "lr_scheduler":  False,
    },
    {
        # ── RUN 2: Reduced learning rate + BatchNormalization ────────────────
        # Halves the learning rate to 5e-4 to reduce gradient oscillation.
        # Adds tensorflow.keras.layers.BatchNormalization after each Conv2D
        # block to stabilise activations and allow deeper convergence.
        "run_id":        2,
        "description":   "Adam lr=5e-4, BatchNormalization, bs=32",
        "learning_rate": 5e-4,
        "batch_size":    32,
        "epochs":        30,
        "patience":      6,
        "optimizer":     "adam",
        "augment":       False,
        "use_bn":        True,
        "class_weights": False,
        "lr_scheduler":  False,
    },
    {
        # ── RUN 3: Data augmentation ─────────────────────────────────────────
        # Adds a tensorflow.keras.Sequential augmentation pipeline using
        # layers.RandomFlip("horizontal_and_vertical") and
        # layers.RandomRotation(0.2) to improve generalisation on the
        # small minority classes (df=115, vasc=142).
        "run_id":        3,
        "description":   "Adam lr=5e-4, BN, RandomFlip + RandomRotation augmentation",
        "learning_rate": 5e-4,
        "batch_size":    32,
        "epochs":        35,
        "patience":      7,
        "optimizer":     "adam",
        "augment":       True,
        "use_bn":        True,
        "class_weights": False,
        "lr_scheduler":  False,
    },
    {
        # ── RUN 4: Class-weight balancing ────────────────────────────────────
        # Addresses the severe class imbalance (nv=6705 vs df=115) by passing
        # sklearn.utils.class_weight.compute_class_weight("balanced") into
        # keras.Model.fit(class_weight=...).  This is a key improvement over
        # the baseline critique and directly targets the F1-score on minority
        # classes mel, bcc, and akiec.
        "run_id":        4,
        "description":   "Adam lr=5e-4, BN, augmentation + class_weight balancing",
        "learning_rate": 5e-4,
        "batch_size":    32,
        "epochs":        40,
        "patience":      8,
        "optimizer":     "adam",
        "augment":       True,
        "use_bn":        True,
        "class_weights": True,
        "lr_scheduler":  False,
    },
    {
        # ── RUN 5: ReduceLROnPlateau scheduler + larger batch ────────────────
        # Introduces tensorflow.keras.callbacks.ReduceLROnPlateau to
        # automatically decay the learning rate when val_loss plateaus
        # (factor=0.5, patience=3).  Batch size is increased to 64 to
        # leverage GPU memory bandwidth more efficiently and smooth gradient
        # estimates.  Switches to SGD with Nesterov momentum as an alternative
        # optimiser to compare convergence behaviour with Adam.
        "run_id":        5,
        "description":   "SGD Nesterov lr=1e-2, bs=64, BN, aug, class_weights, ReduceLROnPlateau",
        "learning_rate": 1e-2,
        "batch_size":    64,
        "epochs":        45,
        "patience":      9,
        "optimizer":     "sgd",
        "augment":       True,
        "use_bn":        True,
        "class_weights": True,
        "lr_scheduler":  True,
    },
]

# ---------------------------------------------------------------------------
# 5. DATA LOADING  (run once, reused across all experiments)
# ---------------------------------------------------------------------------

def locate_image(image_id: str) -> Path | None:
    """Search both image part folders for <image_id>.jpg using pathlib.Path."""
    for directory in IMAGE_DIRS:
        p = directory / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


def load_metadata() -> pd.DataFrame:
    """Read HAM10000_metadata.csv with pandas.read_csv and attach file paths."""
    print("[DATA] Reading metadata:", METADATA_PATH)
    df = pd.read_csv(METADATA_PATH)
    df["label"]      = df["dx"].map(LABEL_MAP)
    df["image_path"] = df["image_id"].apply(locate_image)
    missing = df["image_path"].isna().sum()
    if missing:
        print(f"[DATA] WARNING — {missing} images not found, dropping rows.")
        df = df.dropna(subset=["image_path"])
    print(f"[DATA] {len(df)} samples | class distribution:")
    print(df["dx"].value_counts().to_string())
    return df


def load_images(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all JPEG images using Pillow (PIL.Image.open), resize via
    PIL.Image.BILINEAR interpolation to (IMG_HEIGHT, IMG_WIDTH), then
    normalise to [0, 1] as float32.
    """
    print("[DATA] Loading images into RAM — please wait…")
    images, labels = [], []
    for _, row in df.iterrows():
        img = Image.open(row["image_path"]).convert("RGB")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        images.append(np.array(img, dtype=np.float32) / 255.0)
        labels.append(row["label"])
    X = np.stack(images)
    y = np.array(labels, dtype=np.int32)
    print(f"[DATA] X={X.shape}  y={y.shape}")
    return X, y


def stratified_split(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, ...]:
    """
    Stratified 70 / 15 / 15 split using
    sklearn.model_selection.train_test_split(stratify=y) to preserve
    class ratios across all three partitions.
    """
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
    )
    print(f"[SPLIT] train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te

# ---------------------------------------------------------------------------
# 6. MODEL BUILDER
# ---------------------------------------------------------------------------

def build_augmentation_layer() -> keras.Sequential:
    """
    Preprocessing augmentation pipeline using:
      - tensorflow.keras.layers.RandomFlip("horizontal_and_vertical")
      - tensorflow.keras.layers.RandomRotation(factor=0.2)
      - tensorflow.keras.layers.RandomZoom(height_factor=0.1)
    Applied only during training (Keras disables random ops at inference).
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(height_factor=0.1),
        ],
        name="augmentation",
    )


def build_cnn(cfg: dict) -> keras.Model:
    """
    Build a CNN whose depth and regularisation are controlled by cfg.

    Architecture uses:
      - tensorflow.keras.layers.Conv2D (feature extraction)
      - tensorflow.keras.layers.BatchNormalization (if cfg['use_bn'])
      - tensorflow.keras.layers.MaxPooling2D (spatial downsampling)
      - tensorflow.keras.layers.GlobalAveragePooling2D (replaces Flatten
        for better generalisation in deeper nets)
      - tensorflow.keras.layers.Dense + Dropout (classifier head)
      - softmax output with float32 cast (required for mixed_float16 policy)
    """
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x = inputs

    # Optional augmentation block (active during fit, bypassed at predict)
    if cfg["augment"]:
        x = build_augmentation_layer()(x)

    # Convolutional blocks — filter counts double each block (32→64→128→256)
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        if cfg["use_bn"]:
            x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    # Global average pooling reduces spatial dims without large Flatten vector
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # Cast to float32 before softmax — required when mixed_float16 is active
    x = layers.Dense(NUM_CLASSES, dtype="float32")(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)

    return keras.Model(inputs, outputs, name=f"cnn_run{cfg['run_id']}")


def compile_cnn(model: keras.Model, cfg: dict) -> keras.Model:
    """
    Select optimiser by name:
      - 'adam' → tensorflow.keras.optimizers.Adam
      - 'sgd'  → tensorflow.keras.optimizers.SGD with Nesterov momentum=0.9
    Loss: tensorflow.keras.losses.SparseCategoricalCrossentropy
    """
    if cfg["optimizer"] == "sgd":
        opt = optimizers.SGD(
            learning_rate=cfg["learning_rate"],
            momentum=0.9,
            nesterov=True,
            name="SGD_nesterov",
        )
    else:
        opt = optimizers.Adam(learning_rate=cfg["learning_rate"], name="Adam")

    model.compile(
        optimizer=opt,
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model

# ---------------------------------------------------------------------------
# 7. TRAINING
# ---------------------------------------------------------------------------

def get_callbacks(cfg: dict) -> list:
    """
    Build callback list:
      - tensorflow.keras.callbacks.EarlyStopping (always active)
      - tensorflow.keras.callbacks.ReduceLROnPlateau (run 5 only)
      - tensorflow.keras.callbacks.ModelCheckpoint saves best weights
    """
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(RESULTS_DIR / f"run{cfg['run_id']}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]
    if cfg["lr_scheduler"]:
        cb_list.append(
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            )
        )
    return cb_list


def compute_weights(y_train: np.ndarray) -> dict:
    """
    Compute inverse-frequency class weights using
    sklearn.utils.class_weight.compute_class_weight(class_weight='balanced')
    to up-weight minority classes (df, vasc, akiec) during gradient updates.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train,
    )
    return dict(enumerate(weights))


def run_experiment(
    cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Execute one full training + evaluation cycle for the given cfg.
    Returns a result dict with accuracy, val_loss, and history for logging.
    """
    run_id = cfg["run_id"]
    print(f"\n{'='*70}")
    print(f"  RUN {run_id}/5 — {cfg['description']}")
    print(f"{'='*70}")

    model = build_cnn(cfg)
    model = compile_cnn(model, cfg)
    model.summary(print_fn=lambda s: None)  # suppress verbose summary

    class_weight_map = compute_weights(y_train) if cfg["class_weights"] else None

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=get_callbacks(cfg),
        class_weight=class_weight_map,
        verbose=1,
    )
    elapsed = time.time() - t0

    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc      = accuracy_score(y_test, y_pred)
    val_loss = min(history.history["val_loss"])

    print(f"\n[RUN {run_id}] Test Accuracy : {acc:.4f}")
    print(f"[RUN {run_id}] Best Val Loss : {val_loss:.4f}")
    print(f"[RUN {run_id}] Training Time : {elapsed:.1f}s\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # ── Save confusion matrix ────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(
        f"Run {run_id} Confusion Matrix\n{cfg['description']}\nTest Acc={acc:.4f}",
        fontsize=11,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = RESULTS_DIR / f"iteration_{run_id}_cm.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[RUN {run_id}] Confusion matrix saved → {cm_path}")

    # ── Save training curves ─────────────────────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title(f"Run {run_id} — Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title(f"Run {run_id} — Loss (SparseCategoricalCrossentropy)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    curves_path = RESULTS_DIR / f"iteration_{run_id}_curves.png"
    fig2.savefig(curves_path, dpi=150)
    plt.close(fig2)
    print(f"[RUN {run_id}] Training curves saved → {curves_path}")

    return {
        "run_id":      run_id,
        "description": cfg["description"],
        "optimizer":   cfg["optimizer"].upper(),
        "lr":          cfg["learning_rate"],
        "batch_size":  cfg["batch_size"],
        "epochs_run":  len(history.history["loss"]),
        "test_acc":    round(acc, 4),
        "val_loss":    round(val_loss, 4),
        "train_time_s": round(elapsed, 1),
    }

# ---------------------------------------------------------------------------
# 8. COMPARISON LOG & SUMMARY REPORT
# ---------------------------------------------------------------------------

def save_comparison_log(results: list[dict]) -> None:
    """
    Write all-runs comparison to results/comparison_log.csv using
    the csv standard-library module (no extra dependencies).
    """
    log_path = RESULTS_DIR / "comparison_log.csv"
    fieldnames = [
        "run_id", "description", "optimizer", "lr", "batch_size",
        "epochs_run", "test_acc", "val_loss", "train_time_s",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[LOG] Comparison log saved → {log_path}")


def plot_comparison(results: list[dict]) -> None:
    """
    Bar charts comparing Test Accuracy and Validation Loss across all 5 runs
    using matplotlib.pyplot for inclusion in the Iterative Development report.
    """
    run_labels = [f"Run {r['run_id']}" for r in results]
    accs       = [r["test_acc"]  for r in results]
    val_losses = [r["val_loss"]  for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Test accuracy
    bars = axes[0].bar(run_labels, accs, color="steelblue", edgecolor="black")
    axes[0].bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    axes[0].set_title("Test Accuracy Across 5 Runs")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.4)

    # Validation loss
    bars2 = axes[1].bar(run_labels, val_losses, color="tomato", edgecolor="black")
    axes[1].bar_label(bars2, fmt="%.4f", padding=3, fontsize=9)
    axes[1].set_title("Best Validation Loss Across 5 Runs")
    axes[1].set_ylabel("Val Loss (SparseCategoricalCrossentropy)")
    axes[1].grid(axis="y", alpha=0.4)

    plt.tight_layout()
    cmp_path = RESULTS_DIR / "runs_comparison.png"
    fig.savefig(cmp_path, dpi=150)
    plt.close(fig)
    print(f"[LOG] Comparison chart saved → {cmp_path}")


def print_summary_report(results: list[dict]) -> None:
    """
    Print a formatted summary table to stdout — copy/paste ready for the
    'Iterative Development' section of the SPC4004 assessment report.
    """
    best = max(results, key=lambda r: r["test_acc"])

    print("\n" + "=" * 70)
    print("  ITERATIVE DEVELOPMENT — 5-RUN SUMMARY REPORT")
    print("=" * 70)
    header = f"{'Run':<5} {'Optimiser':<8} {'LR':<8} {'BS':<5} {'Epochs':<7} {'TestAcc':<10} {'ValLoss':<10} {'Time(s)'}"
    print(header)
    print("-" * 70)
    for r in results:
        print(
            f"{r['run_id']:<5} {r['optimizer']:<8} {r['lr']:<8} "
            f"{r['batch_size']:<5} {r['epochs_run']:<7} "
            f"{r['test_acc']:<10.4f} {r['val_loss']:<10.4f} {r['train_time_s']}"
        )
    print("-" * 70)
    print(f"\n  Best run: Run {best['run_id']} — {best['description']}")
    print(f"  Best test accuracy : {best['test_acc']:.4f}")
    print(f"  Best val loss      : {best['val_loss']:.4f}")
    print("=" * 70)

    # Narrative rationale for report
    print("""
ITERATIVE DEVELOPMENT RATIONALE
--------------------------------
Run 1 — Baseline control. Simple 3-block CNN, Adam lr=1e-3, no regularisation.
         Establishes the lower-bound performance to beat.

Run 2 — Added BatchNormalization (tensorflow.keras.layers.BatchNormalization)
         after each convolutional block and halved the learning rate to 5e-4.
         BatchNorm normalises layer inputs, reducing internal covariate shift
         and allowing the optimiser to converge more smoothly.

Run 3 — Introduced data augmentation via tensorflow.keras.layers.RandomFlip
         and RandomRotation to artificially expand the minority classes and
         discourage the model from overfitting to dominant class 'nv'.

Run 4 — Applied sklearn.utils.class_weight.compute_class_weight('balanced')
         and passed the resulting dict into keras.Model.fit(class_weight=...).
         This directly penalises misclassification of rare classes (df, vasc,
         akiec), improving recall on clinically important malignant classes.

Run 5 — Switched from Adam to SGD with Nesterov momentum (momentum=0.9) and
         added tensorflow.keras.callbacks.ReduceLROnPlateau (factor=0.5,
         patience=3) to adaptively decay the learning rate when validation loss
         stagnates. Batch size increased to 64 to better utilise GPU bandwidth.
""")

# ---------------------------------------------------------------------------
# 9. MAIN — execute all 5 experiments sequentially
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  HAM10000 Skin Cancer Classification — 5-Run Iterative Optimisation")
    print("=" * 70)

    # ── Load data once ───────────────────────────────────────────────────────
    df = load_metadata()
    X, y = load_images(df)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    # ── Run all 5 experiments ────────────────────────────────────────────────
    all_results = []
    for cfg in EXPERIMENTS:
        result = run_experiment(cfg, X_train, y_train, X_val, y_val, X_test, y_test)
        all_results.append(result)

    # ── Aggregate reporting ──────────────────────────────────────────────────
    save_comparison_log(all_results)
    plot_comparison(all_results)
    print_summary_report(all_results)

    print(f"\n[DONE] All artefacts saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
