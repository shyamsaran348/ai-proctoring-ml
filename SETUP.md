# Detailed Setup and Run Guide

This guide provides step-by-step instructions for your team members to set up, run, and test the **Temporal Behavioral Inference Engine (TBIE) AI Proctoring Platform** locally.

## Prerequisites
- **Operating System:** Linux, macOS, or Windows (WSL2 recommended for native performance).
- **Python Version:** Python 3.10 or higher.
- **Webcam:** Required for testing the live proctoring features.

## Step 1: Extract the Project
Unzip the provided project archive into your desired workspace directory.
```bash
unzip ai-proctoring-ml.zip
cd ai-proctoring-ml
```

## Step 2: Virtual Environment Setup
It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts with your system's Python packages.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

## Step 3: Install Dependencies
The project relies on specific versions of ML packages (PyTorch, OpenCV) and Django. 
Because OpenCV headless is required to avoid UI-thread binding issues on servers, run the following:

```bash
# Provide base PyTorch depending on your system 
# (CPU-only version is faster to download and sufficient for inference testing)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
pip install opencv-python-headless numpy pandas PyYAML django djangorestframework django-cors-headers python-dotenv mongoengine
```

## Step 4: Verify Machine Learning Weights
Before starting the backend, verify that the `proctoring_ml_module/models/` directory contains the required pre-trained PyTorch weight files:
- `uc1_resnet_embedder.pth`
- `uc2_lstm.pth`
- `presence_model.pth`
- `uc4_drift_model.pth`
- `gam_model.pth`
- `hgdm_model.pth`
- `uc5_risk_gru_v3.pth`

*(If missing, you can regenerate them by running the respective `train_*.py` scripts located in the `/ml` directory).*

## Step 5: Database Initialization (Django)
The project uses SQLite by default, making setup seamless without requiring an external database server.

```bash
# Navigate to the Django root
cd legacy_system

# Create database tables
python manage.py makemigrations
python manage.py migrate

# Create an administrator account (Required to access the Faculty Dashboard)
python manage.py createsuperuser 
# (Follow the prompts to set username, email, and password)
```

## Step 6: Running the Server
Start the Django development server:

```bash
python manage.py runserver 0.0.0.0:8000
```
*Note: Due to browser security policies regarding webcam access (`getUserMedia`), you must access the platform via `http://localhost:8000` or `http://127.0.0.1:8000`. Accessing via a raw IP over HTTP may block webcam access.*

## Step 7: Testing the Flows

### As a Student (Proctoring Client)
1. Open a browser and navigate to `http://localhost:8000/`.
2. Register a new student account or log in.
3. You will be prompted with the **Biometric Onboarding Wizard**. Allow webcam access, align your face, and click **Enroll**.
4. Start an exam. Complete the **Identity MFA** (Face Match -> ID Card Scan).
5. The exam will start. The AI will monitor your movements locally and stream metrics to the backend.

### As Faculty (Admin Dashboard)
1. Open an incognito window or distinct browser and log in at `http://localhost:8000/admin/` using the superuser created in Step 5.
2. Navigate to `http://localhost:8000/faculty/` to access the **Proctoring Command Center**.
3. You will see real-time SSE alerts pop up if the student (from your other window) exhibits anomalous behavior.
4. Click **Deep Dive** on a student row to test the real-time **Intervention Warning System**, and view the **Black Box Evidence Gallery**.

## Troubleshooting
- **No Webcam Detected:** Ensure no other application (like Zoom) is holding an exclusive lock on the camera.
- **IndentationError / Import Errors:** Ensure the `PYTHONPATH` includes the `legacy_system` directory if you are running test scripts outside `manage.py`.
```bash
export DJANGO_SETTINGS_MODULE=coding_exam_system.settings
export PYTHONPATH=$PYTHONPATH:$(pwd)/legacy_system
```
