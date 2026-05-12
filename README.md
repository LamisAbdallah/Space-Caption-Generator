# 🌌 Space Intelligence Platform

A comprehensive, dual-pipeline AI platform engineered for advanced astronomical data analysis. This project features a Custom Web UI (Flask) powered by an asynchronous Deep Learning Backend (FastAPI).

## 🚀 Features

1. **Deep Space Vision Search (VLM Pipeline)**
   * Powered by **SigLIP** (`google/siglip2-so400m-patch16-384`).
   * **Text-to-Image**: Search for deep space objects using natural language.
   * **Image-to-Image**: Upload an image to find visually and semantically similar cosmic imagery.
2. **Asteroid Hazard Predictor (NASA NEO Pipeline)**
   * Powered by a **Random Forest Classifier** trained on live NASA NeoWs telemetry data.
   * Predicts if a Near-Earth Object (NEO) is potentially hazardous based on orbital magnitude, diameter, velocity, and miss distance.
   * Includes human-readable SHAP-style explainability for every prediction.

## 🏗️ Architecture

The platform uses a decoupled microservice architecture:
* **Frontend**: Lightweight HTML/CSS/JS interface served via `Flask` (Port `5000`).
* **Backend API**: High-performance asynchronous inference engine using `FastAPI` (Port `8000`).
* **Model Manager**: Custom Singleton pattern ensuring only one intensive Deep Learning model is loaded into VRAM at a given time.

---

## 💻 How to Run Locally

You will need two terminal windows running simultaneously to run both the backend API and the frontend UI.

### 1️⃣ Start the Backend (Inference Engine)
Open a terminal and navigate to the `backend` directory:
```bash
cd backend
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start the FastAPI server
python api.py
```
*(The backend will run on `http://127.0.0.1:8000`)*

### 2️⃣ Start the Frontend (Web UI)
Open a **second** terminal and navigate to the `frontend` directory:
```bash
cd frontend
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start the Flask web server
python app.py
```
*(The frontend will run on `http://127.0.0.1:5000`)*

### 3️⃣ View the App
Open your web browser and go to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🧬 Repository Structure

```
├── backend/                  # FastAPI Application & AI Models
│   ├── api.py                # Main backend router & Model Manager
│   ├── dataset/              # Stores the VLM and NASA NEO tabular datasets
│   ├── models/               # SigLIP and Random Forest prediction logic
│   └── requirements.txt      # Backend Python dependencies
├── frontend/                 # Flask Application & Web UI
│   ├── app.py                # Flask server routing
│   ├── static/               # CSS and JS (Handles API Fetching)
│   ├── templates/            # HTML structure (index.html)
│   └── requirements.txt      # Frontend Python dependencies (Flash & Werkzeug)
└── README.md                 # Project Documentation
```

## ✨ Created By
**Pink Algorithm Team**
