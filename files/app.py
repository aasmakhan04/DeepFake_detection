import os
import uuid
import json
import numpy as np
import re
import sqlite3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, flash, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
IMG_SIZE = (224, 224)
CLASS_NAMES = ["REAL", "AI_GENERATED", "AI_EDITED"]
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
UPLOAD_FOLDER = os.path.join("static", "uploads")
STATIC_FOLDER = "static"
MAX_CONTENT_MB = 80
MAX_IMAGES = 5

MODEL_PATHS = {
    "EfficientNet": "efficientnet_model.h5",
    "ResNet": "resnet_model.h5",
    "Xception": "xception_model.h5",
}

STATIC_METRICS = {
    "EfficientNet": {"accuracy": 0.9650, "precision": 0.9520, "recall": 0.9710, "f1": 0.9614},
    "ResNet": {"accuracy": 0.9380, "precision": 0.9240, "recall": 0.9400, "f1": 0.9319},
    "Xception": {"accuracy": 0.9510, "precision": 0.9460, "recall": 0.9550, "f1": 0.9504},
    "Ensemble": {"accuracy": 0.9820, "precision": 0.9780, "recall": 0.9850, "f1": 0.9814},
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "visionforge-secret-key"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
# Database & Auth Setup
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn


def create_table():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()


create_table()


def is_valid_password(password):
    if len(password) < 8: return False
    if not re.search(r"[A-Z]", password): return False
    if not re.search(r"[a-z]", password): return False
    if not re.search(r"[0-9]", password): return False
    if not re.search(r"[!@#$%^&*]", password): return False
    return True


# ─────────────────────────────────────────────
# Model Loading & ML Logic
# ─────────────────────────────────────────────
MODELS = {}


def load_model_safe(path: str, name: str):
    if not os.path.isfile(path):
        print(f"[Warning] Model file not found: {path} — {name} will be unavailable.")
        return None
    try:
        model = tf.keras.models.load_model(path, compile=False)
        print(f"[Info] Loaded model: {name}")
        return model
    except Exception as e:
        print(f"[Error] Could not load {name}: {e}")
        return None


for model_name, model_path in MODEL_PATHS.items():
    m = load_model_safe(model_path, model_name)
    if m is not None:
        MODELS[model_name] = m


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(path: str, model_choice: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE, resample=Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    # Dynamic preprocessing based on model
    if model_choice == "EfficientNet" or model_choice == "Ensemble":
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    elif model_choice == "ResNet":
        arr = tf.keras.applications.resnet.preprocess_input(arr)
    elif model_choice == "Xception":
        arr = tf.keras.applications.xception.preprocess_input(arr)

    return arr


def run_single_model(model, image_path: str, model_name: str) -> np.ndarray:
    arr = preprocess_image(image_path, model_name)
    return model.predict(arr, verbose=0)[0]


def run_inference(image_path: str, model_choice: str) -> dict:
    if model_choice == "Ensemble":
        if len(MODELS) > 0:
            preds = []
            for name, m in MODELS.items():
                preds.append(run_single_model(m, image_path, name))
            probs = np.mean(preds, axis=0)
        else:
            raise ValueError("No models loaded for ensemble.")
    else:
        model = MODELS.get(model_choice)
        if model:
            probs = run_single_model(model, image_path, model_choice)
        else:
            raise ValueError(f"{model_choice} is not loaded.")

    class_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[class_idx]
    confidence = float(probs[class_idx])

    return {
        "predicted_class": predicted_class,
        "class_index": class_idx,
        "confidence": round(confidence * 100, 2),
        "all_probabilities": {
            name: round(float(p) * 100, 2)
            for name, p in zip(CLASS_NAMES, probs)
        },
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user" in session: return redirect("/dashboard")
    if request.method == "POST":
        name, email, username, password = request.form["name"], request.form["email"], request.form["username"], \
        request.form["password"]
        if not is_valid_password(password):
            flash("Password must be 8+ chars, include uppercase, lowercase, number, and special character.")
            return redirect("/register")
        hashed_password = generate_password_hash(password)
        try:
            conn = get_db()
            conn.execute("INSERT INTO users (name, email, username, password) VALUES (?, ?, ?, ?)",
                         (name, email, username, hashed_password))
            conn.commit()
            conn.close()
            flash("Registration successful. Please login.")
            return redirect("/login")
        except sqlite3.IntegrityError:
            flash("User already exists. Try different username/email.")
            return redirect("/register")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session: return redirect("/dashboard")
    if request.method == "POST":
        username, password = request.form["username"], request.form["password"]
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        if user is None or not check_password_hash(user["password"], password):
            flash("Incorrect username or password.")
            return redirect("/login")
        session["user"] = user["username"]
        return redirect("/dashboard")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        flash("Please log in to access the dashboard.")
        return redirect("/login")

    available_models = list(MODELS.keys())
    if len(available_models) > 1:
        available_models.append("Ensemble")
    if not available_models:
        available_models = ["No Models Found"]

    return render_template("dashboard.html", available_models=available_models)


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session: return jsonify({"error": "Unauthorized"}), 401

    files = request.files.getlist("files")
    model_choice = request.form.get("model")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files uploaded."}), 400
    if len(files) > MAX_IMAGES:
        return jsonify({"error": f"Maximum {MAX_IMAGES} images allowed."}), 400

    results = []
    for file in files:
        if file.filename == "" or not allowed_file(file.filename): continue

        uid = uuid.uuid4().hex
        ext = file.filename.rsplit(".", 1)[1].lower()
        filename = f"{uid}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        try:
            result = run_inference(save_path, model_choice)
        except Exception as exc:
            result = {
                "predicted_class": "ERROR",
                "confidence": 0,
                "all_probabilities": {"REAL": 0, "AI_GENERATED": 0, "AI_EDITED": 0},
                "error": str(exc),
            }

        result["image_url"] = f"/static/uploads/{filename}"
        result["original_name"] = secure_filename(file.filename)
        results.append(result)

    metrics = STATIC_METRICS.get(model_choice, STATIC_METRICS["EfficientNet"])

    return jsonify({"results": results, "model": model_choice, "metrics": metrics})


@app.route("/static/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)