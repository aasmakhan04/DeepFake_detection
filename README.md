# 👁️ DeepShield: AI Deepfake Detection System
**Final Year Major Project** 🎓

Hello! Welcome to the repository for my major project, **DeepShield**. 

This is a web-based application I built to detect whether an image is real, AI-generated, or digitally edited. As generative AI becomes more advanced, it is getting harder to trust digital media. My goal with this project was to learn how to apply Deep Learning and Computer Vision to solve this real-world problem.

---

## 🚀 What This Project Does
* **Analyzes Images:** You can drag and drop images into the dashboard, and the system will scan them for hidden AI artifacts.
* **Compares Neural Networks:** I integrated three different machine learning models so users can see how different architectures perform.
* **Ensemble Learning:** I built an "Ensemble" feature that combines the predictions of all the models to give one highly accurate final verdict.
* **Visualizes Data:** The results are displayed using interactive charts (built with Chart.js) to show the exact probability breakdown.

---

## 💻 Technology Stack I Used
To build this full-stack application, I used:
* **Backend:** Python, Flask, SQLite (for user login/registration)
* **Machine Learning:** TensorFlow, Keras, NumPy, Pillow
* **Frontend:** HTML5, CSS3 (Glassmorphism design), Vanilla JavaScript
* **Data Visualization:** Chart.js

---

## 🧠 The Machine Learning Models
The core of this project relies on Convolutional Neural Networks (CNNs). I trained and evaluated the models using a subset of the Celeb-DF dataset. The models included in this project are:
1. `EfficientNet B0`
2. `ResNet 50`
3. `Xception`

---

## ⚙️ How to Run This Project Locally

**Step 1: Clone the repository**

  Bash
  git clone https://github.com/YourUsername/deepfake_detection.git

**Step 2: Set up a virtual environment**

  Bash
  python -m venv venv
  Activate on Windows:
  venv\Scripts\activate

**Step 3: Install the required libraries**

  Bash
  pip install flask tensorflow pillow numpy werkzeug

**Step 4: Add the Models**

Place your trained efficientnet_model.h5, resnet_model.h5, and xception_model.h5 files directly into the main project folder (right next to app.py).

**Step 5: Start the server**

  Bash
  python app.py
Finally, open your web browser and go to http://127.0.0.0.1 to view the dashboard!

## 📁 Project Structure
app.py - The main Python backend server.

users.db - The SQLite database that stores user login info.

templates/ - Contains the HTML files (index.html, login.html, etc.).

static/uploads/ - Temporary folder where uploaded images are saved for analysis.

##Created by Aasma Khan
