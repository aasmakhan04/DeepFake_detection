import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4, Xception, ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

CLASS_NAMES = ["real", "ai_generated", "ai_edited"]
DATASET_DIR = "dataset"

AUTOTUNE = tf.data.AUTOTUNE

# =========================
# DATA PIPELINE
# =========================
def load_dataset(split):
    path = os.path.join(DATASET_DIR, split)

    return tf.keras.utils.image_dataset_from_directory(
        path,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        seed=42
    )

data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

def preprocess(x, y):
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x, y

def build_pipeline(split):
    ds = load_dataset(split)

    if split == "train":
        ds = ds.map(lambda x,y: (data_aug(x), y), num_parallel_calls=AUTOTUNE)

    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.cache().prefetch(AUTOTUNE)

    return ds

train_ds = build_pipeline("train")
val_ds   = build_pipeline("valid")
test_ds  = build_pipeline("test")

# =========================
# CLASS WEIGHTS
# =========================
def compute_class_weights(ds):
    counts = np.zeros(len(CLASS_NAMES))

    for _, labels in ds:
        y = np.argmax(labels.numpy(), axis=1)
        for i in y:
            counts[i] += 1

    total = np.sum(counts)

    return {
        i: total / (len(CLASS_NAMES) * counts[i])
        for i in range(len(CLASS_NAMES))
    }

class_weights = compute_class_weights(train_ds)

# =========================
# MODEL BUILDER
# =========================
def build_model(name):

    if name == "efficientnet":
        base = EfficientNetB4(include_top=False, weights="imagenet", input_shape=(224,224,3))
    elif name == "xception":
        base = Xception(include_top=False, weights="imagenet", input_shape=(224,224,3))
    elif name == "resnet":
        base = ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))

    base.trainable = False

    inputs = tf.keras.Input(shape=(224,224,3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(3, activation="softmax")(x)

    return Model(inputs, outputs), base

# =========================
# TRAIN + EVALUATE
# =========================
results = {}

def train_and_evaluate(name):

    print(f"\n🚀 Training {name.upper()}")

    model, base = build_model(name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(f"{name}_model.h5", save_best_only=True),
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=2)
    ]

    # Stage 1
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )

    # Fine-tune
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        class_weight=class_weights
    )

    # =========================
    # EVALUATION
    # =========================
    y_true, y_pred = [], []

    for x, y in test_ds:
        pred = model.predict(x)
        y_true.extend(tf.argmax(y, axis=1))
        y_pred.extend(tf.argmax(pred, axis=1))

    report = classification_report(y_true, y_pred, output_dict=True)

    results[name] = {
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"]
    }

    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(name)
    plt.savefig(f"{name}_cm.png")
    plt.close()

# =========================
# RUN ALL MODELS
# =========================
for model_name in ["efficientnet", "xception", "resnet"]:
    train_and_evaluate(model_name)

# =========================
# PLOT COMPARISON
# =========================
models = list(results.keys())

acc = [results[m]["accuracy"] for m in models]
prec = [results[m]["precision"] for m in models]
rec = [results[m]["recall"] for m in models]

plt.figure()
plt.plot(models, acc, marker='o', label="Accuracy")
plt.plot(models, prec, marker='o', label="Precision")
plt.plot(models, rec, marker='o', label="Recall")
plt.legend()
plt.title("Model Comparison")
plt.savefig("comparison.png")
plt.show()

print("\n✅ Training Complete!")