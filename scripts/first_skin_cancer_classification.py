"""
SPC4004 Assessment 2 - Skin Cancer Classification (HAM10000)
=============================================================
Multi-class classification of 7 skin lesion types using a baseline
Convolutional Neural Network (CNN) built with TensorFlow/Keras.

Dataset: HAM10000 (Human Against Machine with 10000 training images)
Classes: akiec, bcc, bkl, df, mel, nv, vasc
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, callbacks

# ---------------------------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------------------------
# Use pathlib.Path for cross-platform, relative path management
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIRS = [
    BASE_DIR / "HAM10000_images_part_1",
    BASE_DIR / "HAM10000_images_part_2",
]
METADATA_PATH = BASE_DIR / "HAM10000_metadata.csv"

# Image preprocessing parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001

# Reproducibility seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# HAM10000 diagnostic label mapping (7 classes)
LABEL_MAP = {
    "akiec": 0,  # Actinic keratoses
    "bcc": 1,    # Basal cell carcinoma
    "bkl": 2,    # Benign keratosis-like lesions
    "df": 3,     # Dermatofibroma
    "mel": 4,    # Melanoma
    "nv": 5,     # Melanocytic nevi
    "vasc": 6,   # Vascular lesions
}

# Clinical grouping for binary interpretation
MALIGNANT_LABELS = {"akiec", "bcc", "mel"}
BENIGN_LABELS = {"bkl", "df", "nv", "vasc"}

NUM_CLASSES = len(LABEL_MAP)

# ---------------------------------------------------------------------------
# 3. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------------

def locate_image(image_id: str) -> Path | None:
    """Search both image directories for a given image_id (.jpg)."""
    for directory in IMAGE_DIRS:
        path = directory / f"{image_id}.jpg"
        if path.exists():
            return path
    return None


def load_metadata() -> pd.DataFrame:
    """Load HAM10000_metadata.csv using pandas and map labels to integers."""
    print("[INFO] Loading metadata from:", METADATA_PATH)
    df = pd.read_csv(METADATA_PATH)
    df["label"] = df["dx"].map(LABEL_MAP)
    df["image_path"] = df["image_id"].apply(locate_image)
    # Drop any rows where the image file was not found
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"[WARNING] {missing} images not found on disk — dropping rows.")
        df = df.dropna(subset=["image_path"])
    print(f"[INFO] Loaded {len(df)} samples across {NUM_CLASSES} classes.")
    print(df["dx"].value_counts())
    return df


def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    """
    Load a single JPEG image using Pillow (PIL), resize it with
    PIL.Image.resize, and normalise pixel values to [0, 1].
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def prepare_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Load all images into a NumPy array alongside their integer labels."""
    print("[INFO] Loading and preprocessing images — this may take a few minutes...")
    images = []
    labels = []
    for _, row in df.iterrows():
        img = load_and_preprocess_image(row["image_path"])
        images.append(img)
        labels.append(row["label"])
    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    print(f"[INFO] Image array shape: {X.shape}, Labels shape: {y.shape}")
    return X, y


# ---------------------------------------------------------------------------
# 4. STRATIFIED DATA SPLITTING
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified split using sklearn.model_selection.train_test_split
    to maintain class ratios:
        Training   — 70%
        Validation — 15%
        Test       — 15%
    """
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    # Second split: 50% of temp → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )
    print(f"[INFO] Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# 5. MODEL ARCHITECTURE — Baseline CNN
# ---------------------------------------------------------------------------

def build_baseline_cnn() -> keras.Model:
    """
    Construct a baseline Convolutional Neural Network using
    tensorflow.keras.models.Sequential with:
      - tensorflow.keras.layers.Conv2D for feature extraction
      - tensorflow.keras.layers.MaxPooling2D for spatial down-sampling
      - tensorflow.keras.layers.Flatten to vectorise feature maps
      - tensorflow.keras.layers.Dense for classification
      - tensorflow.keras.layers.Dropout for basic regularisation

    NOTE (intentional limitation): This is a deliberately simple architecture
    to serve as a starting point for iterative development. It does not use
    batch normalisation, data augmentation, or transfer learning.
    """
    model = models.Sequential(name="baseline_cnn")

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Classifier head
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    return model


def compile_model(model: keras.Model) -> keras.Model:
    """
    Compile the model with:
      - Cost function: tensorflow.keras.losses.SparseCategoricalCrossentropy
        (suitable for integer-encoded labels).
      - Optimizer: tensorflow.keras.optimizers.Adam with a configurable
        learning rate.
      - Metric: accuracy.

    NOTE (intentional limitation): No class-weight balancing is applied here.
    The HAM10000 dataset is heavily imbalanced (class 'nv' dominates). This
    is left as a known issue for the Critique section of the report.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ---------------------------------------------------------------------------
# 6. TRAINING
# ---------------------------------------------------------------------------

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> keras.callbacks.History:
    """
    Train the model using keras.Model.fit with an EarlyStopping callback
    from tensorflow.keras.callbacks to prevent excessive overfitting.
    """
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


# ---------------------------------------------------------------------------
# 7. EVALUATION & METRICS
# ---------------------------------------------------------------------------

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Evaluate the trained model on the unseen test set using:
      - sklearn.metrics.accuracy_score
      - sklearn.metrics.confusion_matrix
      - sklearn.metrics.classification_report
    """
    # Generate predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"TEST SET ACCURACY: {acc:.4f}")
    print(f"{'='*60}\n")

    # Per-class classification report
    class_names = list(LABEL_MAP.keys())
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("Classification Report:\n")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n")
    print(cm)

    # Plot confusion matrix using seaborn.heatmap and matplotlib
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix — Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
    plt.show()
    print("[INFO] Confusion matrix saved to confusion_matrix.png")


# ---------------------------------------------------------------------------
# 8. VISUALISATION — Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: keras.callbacks.History) -> None:
    """
    Plot training vs. validation accuracy and loss curves using
    matplotlib.pyplot to diagnose overfitting/underfitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curves
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Model Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(True)

    # Loss curves
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Model Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (Sparse Categorical Crossentropy)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "training_curves.png", dpi=150)
    plt.show()
    print("[INFO] Training curves saved to training_curves.png")


# ---------------------------------------------------------------------------
# 9. MAIN EXECUTION
# ---------------------------------------------------------------------------

def main():
    """End-to-end pipeline: load → split → build → train → evaluate."""
    print("=" * 60)
    print("HAM10000 Skin Cancer Classification — Baseline CNN")
    print("=" * 60)

    # Step 1 — Load metadata and images
    df = load_metadata()
    X, y = prepare_dataset(df)

    # Step 2 — Stratified split (70/15/15)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Step 3 — Build and compile baseline CNN
    model = build_baseline_cnn()
    model = compile_model(model)

    # Step 4 — Train with early stopping
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Step 5 — Evaluate on unseen test set
    evaluate_model(model, X_test, y_test)

    # Step 6 — Plot training curves
    plot_training_curves(history)

    print("\n[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()
