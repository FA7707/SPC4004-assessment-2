"""
SPC4004 Assessment 2 — Final Iteration (Run 6)
================================================
High-stability, high-generalisation CNN for HAM10000 skin cancer classification.

Key design decisions documented with named library/technique references:

1. DATA PIPELINE  — tf.data.Dataset.from_tensor_slices + .batch + .prefetch
                     replaces raw NumPy array feeding, capping RAM usage and
                     enabling asynchronous CPU→GPU transfer via tf.data.AUTOTUNE.

2. AUGMENTATION   — tensorflow.keras.layers.RandomFlip,
                     tensorflow.keras.layers.RandomRotation,
                     tensorflow.keras.layers.RandomZoom,
                     tensorflow.keras.layers.RandomContrast
                     (applied inside the model graph so they are GPU-accelerated
                     during training and automatically disabled at inference).

3. REGULARISATION — tensorflow.keras.regularizers.L2 (weight_decay=1e-4) on
                     every Conv2D and Dense kernel, combined with
                     tensorflow.keras.layers.Dropout(0.5) in the classifier head,
                     to suppress the memorisation/overfitting observed in Runs 1-5.

4. DYNAMIC LR     — tensorflow.keras.callbacks.ReduceLROnPlateau monitors
                     val_loss with factor=0.5 and patience=4.  When the
                     validation loss plateaus or jitters, the callback halves the
                     learning rate, enabling fine-grained convergence in later
                     epochs without manual tuning.

5. BEST-MODEL
   RECOVERY       — tensorflow.keras.callbacks.ModelCheckpoint with
                     save_best_only=True and monitor='val_loss' persists only the
                     weights from the epoch with the absolute lowest val_loss.
                     Combined with tensorflow.keras.callbacks.EarlyStopping
                     (restore_best_weights=True), the final in-memory model
                     always reflects that best epoch rather than a noisy late one.

6. EARLY STOP     — tensorflow.keras.callbacks.EarlyStopping monitors val_loss
                     with patience=8.  If the generalisation gap (train_loss
                     dropping while val_loss climbs) persists for 8 consecutive
                     epochs, training is terminated to prevent further overfitting.

7. CLASS BALANCE  — sklearn.utils.class_weight.compute_class_weight('balanced')
                     feeds inverse-frequency weights into keras.Model.fit
                     (class_weight=...) to up-weight minority classes
                     (df=115, vasc=142, akiec=327).

8. OPTIMISER      — tensorflow.keras.optimizers.Adam (lr=3e-4) chosen for its
                     adaptive per-parameter learning rates.  Lower initial LR
                     than Run 1 (1e-3) reduces early gradient oscillation; the
                     ReduceLROnPlateau callback handles fine-tuning from there.

9. ARCHITECTURE   — 5-block CNN (32→64→128→256→512 filters) with
                     tensorflow.keras.layers.BatchNormalization after every
                     Conv2D, terminated by GlobalAveragePooling2D → Dense(512)
                     → Dropout(0.5) → Dense(7, softmax).  Deeper than Runs 1-5
                     to give the model more capacity while regularisation
                     prevents it from memorising.

10. GPU/CUDA      — tf.config.experimental.set_memory_growth prevents TF from
                     claiming all VRAM.  tf.keras.mixed_precision (mixed_float16)
                     is enabled when a GPU is detected, halving memory per
                     activation and accelerating matmul on Tensor Cores.
"""

# ===========================================================================
# 1. IMPORTS
# ===========================================================================
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import (
    layers,
    models,
    optimizers,
    losses,
    callbacks,
    regularizers,
)

# ===========================================================================
# 2. GPU / CUDA CONFIGURATION
# ===========================================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] {len(gpus)} GPU(s) detected — enabling mixed_float16 policy")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
else:
    print("[GPU] No GPU found — training will run on CPU (slower).")

# ===========================================================================
# 3. CONFIGURATION
# ===========================================================================
BASE_DIR      = Path(__file__).resolve().parent
IMAGE_DIRS    = [
    BASE_DIR / "HAM10000_images_part_1",
    BASE_DIR / "HAM10000_images_part_2",
]
METADATA_PATH = BASE_DIR / "HAM10000_metadata.csv"
RESULTS_DIR   = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Image dimensions
IMG_H, IMG_W, IMG_C = 128, 128, 3

# Hyperparameters — Run 6 (final)
LEARNING_RATE  = 3e-4          # Lower than Run-1 (1e-3), avoids oscillation
BATCH_SIZE     = 32            # Moderate — stable gradients on CPU/GPU
EPOCHS         = 60            # High ceiling; EarlyStopping will cut short
PATIENCE_ES    = 8             # EarlyStopping patience (epochs)
PATIENCE_LR    = 4             # ReduceLROnPlateau patience (epochs)
LR_FACTOR      = 0.5           # Halve LR on plateau
MIN_LR         = 1e-6          # Floor for learning rate decay
WEIGHT_DECAY   = 1e-4          # L2 regularisation strength
DROPOUT_RATE   = 0.5           # Classifier head dropout
RANDOM_STATE   = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# HAM10000 diagnostic labels
LABEL_MAP = {
    "akiec": 0,  # Actinic keratoses        (MALIGNANT)
    "bcc":   1,  # Basal cell carcinoma     (MALIGNANT)
    "bkl":   2,  # Benign keratosis-like    (BENIGN)
    "df":    3,  # Dermatofibroma           (BENIGN)
    "mel":   4,  # Melanoma                 (MALIGNANT)
    "nv":    5,  # Melanocytic nevi         (BENIGN)
    "vasc":  6,  # Vascular lesions         (BENIGN)
}
CLASS_NAMES = list(LABEL_MAP.keys())
NUM_CLASSES = len(LABEL_MAP)

# ===========================================================================
# 4. DATA LOADING  (Pillow + NumPy, then wrapped by tf.data)
# ===========================================================================

def locate_image(image_id: str) -> Path | None:
    """Search HAM10000_images_part_1/ and part_2/ for <image_id>.jpg."""
    for d in IMAGE_DIRS:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


def load_metadata() -> pd.DataFrame:
    """Load HAM10000_metadata.csv with pandas.read_csv, attach paths + labels."""
    print(f"[DATA] Loading metadata: {METADATA_PATH}")
    df = pd.read_csv(METADATA_PATH)
    df["label"]      = df["dx"].map(LABEL_MAP)
    df["image_path"] = df["image_id"].apply(locate_image)
    missing = df["image_path"].isna().sum()
    if missing:
        print(f"[DATA] WARNING — {missing} images missing, dropping.")
        df = df.dropna(subset=["image_path"])
    print(f"[DATA] {len(df)} samples loaded.  Class counts:")
    print(df["dx"].value_counts().to_string())
    return df


def load_images(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Load + resize all images using Pillow (PIL.Image.open + .resize with
    PIL.Image.BILINEAR resampling), normalise to [0,1] float32.
    """
    print("[DATA] Reading and resizing JPEG images with Pillow — please wait...")
    imgs, lbls = [], []
    for _, row in df.iterrows():
        img = Image.open(row["image_path"]).convert("RGB")
        img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
        imgs.append(np.array(img, dtype=np.float32) / 255.0)
        lbls.append(row["label"])
    X = np.stack(imgs)
    y = np.array(lbls, dtype=np.int32)
    print(f"[DATA] X.shape={X.shape}  y.shape={y.shape}")
    return X, y


# ===========================================================================
# 5. STRATIFIED SPLIT  (sklearn.model_selection.train_test_split)
# ===========================================================================

def stratified_split(X, y):
    """70/15/15 stratified split preserving class ratios in every partition."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE,
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE,
    )
    print(f"[SPLIT] Train={len(y_tr)}  Val={len(y_val)}  Test={len(y_te)}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ===========================================================================
# 6. tf.data PIPELINE  (RAM-stable, CPU/GPU-friendly)
# ===========================================================================

def make_tf_dataset(X, y, batch_size, shuffle=False):
    """
    Wrap NumPy arrays in a tf.data.Dataset pipeline:
      - tf.data.Dataset.from_tensor_slices  — creates element-wise slices
      - .shuffle(buffer)                    — randomise order each epoch
      - .batch(batch_size)                  — group into mini-batches
      - .prefetch(tf.data.AUTOTUNE)         — overlap CPU prep with GPU compute

    This avoids feeding one giant NumPy array to model.fit(), which can
    cause memory spikes on laptops.  tf.data.AUTOTUNE lets TensorFlow
    dynamically tune the prefetch buffer size.
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), seed=RANDOM_STATE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ===========================================================================
# 7. MODEL ARCHITECTURE — Run 6 Final CNN
# ===========================================================================

def build_augmentation_block():
    """
    In-graph augmentation using Keras preprocessing layers:
      - tensorflow.keras.layers.RandomFlip("horizontal_and_vertical")
      - tensorflow.keras.layers.RandomRotation(factor=0.15)
      - tensorflow.keras.layers.RandomZoom(height_factor=0.1)
      - tensorflow.keras.layers.RandomContrast(factor=0.15)

    These layers are active ONLY during model.fit() (training=True);
    model.predict() / model.evaluate() automatically bypass them.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(height_factor=0.1),
        layers.RandomContrast(factor=0.15),
    ], name="augmentation_block")


def build_final_cnn() -> keras.Model:
    """
    5-block CNN with:
      - tensorflow.keras.layers.Conv2D + tensorflow.keras.regularizers.L2
      - tensorflow.keras.layers.BatchNormalization after each Conv2D
      - tensorflow.keras.layers.MaxPooling2D for spatial reduction
      - tensorflow.keras.layers.GlobalAveragePooling2D (not Flatten)
      - tensorflow.keras.layers.Dense + L2 + Dropout
      - float32 softmax output (safe under mixed_float16 policy)

    Filter progression: 32 → 64 → 128 → 256 → 512
    Total ~3.5 M parameters — larger than Runs 1-5 but still trainable
    on a laptop CPU/GPU in reasonable time.
    """
    l2 = regularizers.L2(WEIGHT_DECAY)

    inputs = keras.Input(shape=(IMG_H, IMG_W, IMG_C), name="image_input")

    # ── Augmentation (training-only) ────────────────────────────────────────
    x = build_augmentation_block()(inputs)

    # ── Conv Block 1: 32 filters ────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2,
                       activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # ── Conv Block 2: 64 filters ────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2,
                       activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # ── Conv Block 3: 128 filters ───────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2,
                       activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # ── Conv Block 4: 256 filters ───────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2,
                       activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # ── Conv Block 5: 512 filters ───────────────────────────────────────────
    x = layers.Conv2D(512, (3, 3), padding="same", kernel_regularizer=l2,
                       activation="relu", name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.MaxPooling2D((2, 2), name="pool5")(x)

    # ── Classifier Head ─────────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=l2,
                     name="fc1")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)

    # float32 output (required for numerical stability under mixed_float16)
    x = layers.Dense(NUM_CLASSES, dtype="float32", name="logits")(x)
    outputs = layers.Activation("softmax", dtype="float32", name="softmax")(x)

    model = keras.Model(inputs, outputs, name="final_cnn_run6")
    return model


# ===========================================================================
# 8. COMPILE + CALLBACKS
# ===========================================================================

def compile_model(model: keras.Model) -> keras.Model:
    """
    Optimiser : tensorflow.keras.optimizers.Adam (lr=3e-4)
    Loss      : tensorflow.keras.losses.SparseCategoricalCrossentropy
    Metric    : accuracy
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, name="Adam"),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def get_callbacks() -> list:
    """
    1. tensorflow.keras.callbacks.EarlyStopping
       — patience=8, restore_best_weights=True
       — Halts training when the generalisation gap widens for 8 epochs.
       — restore_best_weights loads the checkpoint with the lowest val_loss
         back into memory automatically.

    2. tensorflow.keras.callbacks.ReduceLROnPlateau
       — monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6
       — Detects plateau/jitter in val_loss and halves the learning rate,
         allowing progressively finer weight updates.

    3. tensorflow.keras.callbacks.ModelCheckpoint
       — save_best_only=True, monitor='val_loss'
       — Persists only the weights from the single best epoch to disk,
         guaranteeing the saved .keras file reflects the absolute minimum
         val_loss encountered during the entire run.
    """
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_ES,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_FACTOR,
            patience=PATIENCE_LR,
            min_lr=MIN_LR,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=str(RESULTS_DIR / "final_run6_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    sklearn.utils.class_weight.compute_class_weight('balanced')
    returns weights inversely proportional to class frequency.
    Passed to keras.Model.fit(class_weight=...) so the loss function
    penalises errors on rare classes (df, vasc, akiec) more heavily.
    """
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train,
    )
    cw = dict(enumerate(weights))
    print("[CLASS WEIGHTS]", {CLASS_NAMES[k]: round(v, 2) for k, v in cw.items()})
    return cw


# ===========================================================================
# 9. EVALUATION & VISUALISATION
# ===========================================================================

def evaluate_and_report(model, X_test, y_test, history):
    """Generate final confusion matrix, classification report, and plots."""

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)

    # ── Console report ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINAL RUN 6 — TEST ACCURACY: {acc:.4f}")
    print(f"{'='*70}\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # ── Confusion matrix → final_run_cm.png ─────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig1, ax1 = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1)
    ax1.set_title(f"Run 6 (Final) Confusion Matrix — Test Acc = {acc:.4f}",
                  fontsize=12)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = RESULTS_DIR / "final_run_cm.png"
    fig1.savefig(cm_path, dpi=200)
    plt.close(fig1)
    print(f"[SAVED] Confusion matrix → {cm_path}")

    # ── Accuracy & Loss dual-plot → final_performance.png ───────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history.history["accuracy"],     label="Train Accuracy",     linewidth=1.5)
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=1.5)
    axes[0].set_title("Run 6 — Accuracy (train vs val)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train Loss",     linewidth=1.5)
    axes[1].plot(history.history["val_loss"], label="Validation Loss", linewidth=1.5)
    axes[1].set_title("Run 6 — Loss (SparseCategoricalCrossentropy)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    perf_path = RESULTS_DIR / "final_performance.png"
    fig2.savefig(perf_path, dpi=200)
    plt.close(fig2)
    print(f"[SAVED] Performance curves → {perf_path}")

    # ── LR decay plot ───────────────────────────────────────────────────────
    if "lr" in history.history:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(history.history["lr"], linewidth=1.5, color="green")
        ax3.set_title("Run 6 — Learning Rate Decay (ReduceLROnPlateau)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Learning Rate")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        lr_path = RESULTS_DIR / "final_lr_schedule.png"
        fig3.savefig(lr_path, dpi=200)
        plt.close(fig3)
        print(f"[SAVED] LR schedule  → {lr_path}")

    return acc


# ===========================================================================
# 10. MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("  HAM10000 — FINAL ITERATION (Run 6)")
    print("  High-Stability CNN with ReduceLROnPlateau + L2 + Augmentation")
    print("=" * 70)

    # ── Load data ───────────────────────────────────────────────────────────
    df = load_metadata()
    X, y = load_images(df)
    X_tr, X_val, X_te, y_tr, y_val, y_te = stratified_split(X, y)

    # ── Build tf.data pipelines ─────────────────────────────────────────────
    print("[tf.data] Building Dataset pipelines with .batch + .prefetch(AUTOTUNE)")
    ds_train = make_tf_dataset(X_tr, y_tr, BATCH_SIZE, shuffle=True)
    ds_val   = make_tf_dataset(X_val, y_val, BATCH_SIZE)

    # ── Class weights ───────────────────────────────────────────────────────
    cw = compute_class_weights(y_tr)

    # ── Build, compile, summarise ───────────────────────────────────────────
    model = build_final_cnn()
    model = compile_model(model)
    model.summary()

    # ── Train ───────────────────────────────────────────────────────────────
    print(f"\n[TRAIN] Starting Run 6 — max {EPOCHS} epochs, "
          f"EarlyStopping patience={PATIENCE_ES}, "
          f"ReduceLROnPlateau patience={PATIENCE_LR}\n")

    t0 = time.time()
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
        class_weight=cw,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n[TRAIN] Completed in {elapsed:.1f}s  "
          f"({len(history.history['loss'])} epochs)")

    # ── Evaluate on held-out test set ───────────────────────────────────────
    acc = evaluate_and_report(model, X_te, y_te, history)

    # ── Summary for report ──────────────────────────────────────────────────
    best_val = min(history.history["val_loss"])
    final_lr = history.history.get("lr", [LEARNING_RATE])[-1]
    print(f"""
{'='*70}
  RUN 6 SUMMARY — FOR ITERATIVE DEVELOPMENT REPORT
{'='*70}
  Architecture     : 5-block CNN (32→64→128→256→512) + BatchNormalization
  Optimiser        : tensorflow.keras.optimizers.Adam (initial lr={LEARNING_RATE})
  Cost Function    : tensorflow.keras.losses.SparseCategoricalCrossentropy
  Regularisation   : tensorflow.keras.regularizers.L2(l2={WEIGHT_DECAY})
                     + tensorflow.keras.layers.Dropout(rate={DROPOUT_RATE})
  Augmentation     : RandomFlip, RandomRotation, RandomZoom, RandomContrast
  LR Scheduler     : tensorflow.keras.callbacks.ReduceLROnPlateau
                     (factor={LR_FACTOR}, patience={PATIENCE_LR}, min_lr={MIN_LR})
  Early Stopping   : tensorflow.keras.callbacks.EarlyStopping
                     (patience={PATIENCE_ES}, restore_best_weights=True)
  Model Checkpoint : tensorflow.keras.callbacks.ModelCheckpoint
                     (save_best_only=True, monitor='val_loss')
  Class Weighting  : sklearn.utils.class_weight.compute_class_weight('balanced')
  Data Pipeline    : tf.data.Dataset.from_tensor_slices → .batch → .prefetch(AUTOTUNE)

  Epochs trained   : {len(history.history["loss"])}
  Final LR         : {final_lr}
  Best Val Loss    : {best_val:.4f}
  Test Accuracy    : {acc:.4f}
  Training Time    : {elapsed:.1f}s

  Artefacts saved to: {RESULTS_DIR}/
    - final_run_cm.png        (Confusion Matrix, 200 dpi)
    - final_performance.png   (Accuracy + Loss curves, 200 dpi)
    - final_lr_schedule.png   (LR decay plot)
    - final_run6_best.keras   (Best model weights)
{'='*70}
""")


if __name__ == "__main__":
    main()
