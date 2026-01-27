# AI-Powered Proctoring System
**Advanced Identity Verification & Behavioral Analysis Engine**

This repository hosts a hybrid AI-Proctoring solution that combines a robust Django Backend (`legacy_system`) with a state-of-the-art Deep Learning Module (`proctoring_ml_module`) to ensure exam integrity.

---

## 💻 Core Software Features

Beyond the AI, this is a fully functional **Coding Exam Platform**:
1.  **Multi-Language Code Execution:**
    *   Supports **Python** and **JavaScript**.
    *   Runs code safely in sandboxed subprocesses.
    *   Real-time stdout/stderr capture.
2.  **Problem Management:**
    *   Admins can create coding challenges with rich descriptions.
    *   **Test Cases:** Supports Hidden vs. Public test cases for grading.
3.  **Exam Session Engine:**
    *   **Timers:** Auto-submits when time runs out.
    *   **State Saving:** Prevents data loss if the browser crashes.
    *   **Submission History:** Tracks every attempt and score.
4.  **Tech Stack:**
    *   **Backend:** Django 4.2 + Django REST Framework.
    *   **Database:** SQLite (Dev) / PostgreSQL (Prod).
    *   **Frontend:** Vanilla JS + HTML5 (No heavy frameworks required).

---

## 🚀 Key Features (The "UC" Architecture)

The system is built around strict "Use Cases" (UCs) that target specific cheating behaviors.

### 1. UC1: Real-Time Identity Verification (ACTIVE)
**Goal:** Ensure the person taking the exam is 100% the same person who started it.
*   **Model:** **ResNet-50** (Deep Learning via PyTorch).
*   **Mechanism:** One-Shot Enrollment.
*   **Workflow:**
    1.  **Session Start:** The user captures a specialized `session_ref.jpg` (Webcam Selfie).
    2.  **Locking:** This specific image is saved to `media/sessions/{username}/{session_id}/`.
    3.  **Live Monitoring:** Every video frame (approx 5 FPS) is passed to the AI Engine.
    4.  **Comparison:** The AI computes a 256-dimensional embedding vector for the Live Face and compares it to the `session_ref.jpg` embedding using **Cosine Similarity**.
    5.  **Scoring:**
        *   `Sim > 0.65`: **High Confidence Match** (User is verified).
        *   `Sim < 0.40`: **Identity Mismatch** (Warning triggered).
        *   `Sim < 0.20`: **Unknown Person** (Critical Alert).

### 2. UC2: Temporal Instability (Integrated)
**Goal:** Detect nervous, erratic, or unnatural head movements that suggest looking at cheat sheets.
*   **Model:** **LSTM / GRU** (Recurrent Neural Networks).
*   **Logic:** Tracks the "velocity" of head poses over a 60-frame window.
*   **Output:** An "Instability Score" (0.0 to 1.0). High scores indicate suspicious rapid movement.

### 3. UC5: Multi-Modal Risk Fusion (Integrated)
**Goal:** Combine all signals into a single "Cheat Risk" score.
*   **Logic:** It takes inputs from UC1 (Identity) and UC2 (Behavior).
*   **The "Trust" Algorithm:**
    *   **Safety Clamp:** If UC1 Identity Match is very high (> 65%), the system *trusts* the user. It forces the Risk Score down to **0.1 (Safe)**.
    *   **Why?** This prevents false alarms. If we know it's you, we tolerate minor movements.
    *   **Escalation:** If Identity drops or Head Pose becomes extreme (looking away > 15 degrees), the Risk Score escalates immediately.

---

## 📂 Strict Data Organization

To ensure reliability and auditability, we strictly isolate student data.

### 1. Student Profiles
**Location:** `media/students/{username}/`
*   Contains the master `reference.jpg` (ID Card/Profile Photo) used for initial login verification.

### 2. Exam Sessions (The "Clean" Structure)
**Location:** `media/sessions/{username}/{session_id}/`
*   **Isolation:** Every single exam session gets its own folder.
*   **Content:** Contains `session_ref.jpg` (The anchor image for *that specific exam*).
*   **Benefit:** 
    *   We never mix up "Student A" with "Student B".
    *   We never mix up "Exam 1" with "Exam 2".
    *   The AI always compares against the most relevant, recent photo.

---

## 🛠️ Technical Architecture

### 1. The Backend (Legacy System)
*   **Framework:** Django 4.2 + Django REST Framework.
*   **Role:** Manages Users, Exam Problems, Submissions, and API Routing.
*   **Endpoints:**
    *   `/api/sessions/{id}/frame/`: Receives Base64 video frames from the browser.
    *   `/api/register/`: Handles student onboarding with strict photo checks.

### 2. The AI Adapter (`MLProctoringAdapter`)
*   **Role:** Acts as a bridge between Django (Synchronous) and PyTorch (Computational).
*   **File:** `exams/services/ml_adapter.py`.
*   **Function:** 
    1.  Receives the frame from Django.
    2.  Converts Base64 -> NumPy -> Tensor.
    3.  Feeds it to `ProctoringEngine`.
    4.  Returns the `risk`, `similarity`, and `num_faces` metrics.

### 3. The Frontend (Client-Side)
*   **Technology:** Vanilla JavaScript + HTML5 Canvas.
*   **Logic:** 
    *   Captures Webcam feed.
    *   Sends frames to Backend.
    *   Receives Risk Scores.
    *   **Draws Bounding Boxes:**
        *   🟢 **Green:** Low Risk / Verified Identity.
        *   🟠 **Orange:** Warning / Looking Away.
        *   🔴 **Red:** Identity Mismatch / Multi-Face.

---

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure PyTorch and OpenCV are installed)*

2.  **Start the Server:**
    ```bash
    cd legacy_system
    python manage.py runserver
    ```

3.  **Use the System:**
    *   Login as a Student.
    *   Go to **Dashboard**.
    *   **Verify Identity:** Take a snapshot (Saved to `media/students/...`).
    *   **Start Exam:** Take a session snapshot (Saved to `media/sessions/...`).
    *   **Experience Real-Time AI:** The box around your face will update 5 times/second with your real identity score.

---

## 🧪 Verification Logs (Proof of Work)

When running, the server outputs debug logs proving the AI is active:

```text
[Debug] Faces: 1 | Sim: 0.9655 | Risk: 0.1000  <-- High Match (You)
[Debug] Faces: 1 | Sim: 0.8920 | Risk: 0.1000
[Debug] Faces: 0 | Sim: 0.2469 | Risk: 1.0000  <-- Face Left (Cheat Risk!)
```

*   **Sim:** The Cosine Similarity score from the `UC1` Engine.
---

## 🚧 Known Limitations & Future Roadmap

While the Proctoring Engine is advanced, some legacy features are still in development:

1.  **Multiple Choice Questions (MCQ):**
    *   *Current Status:* System only supports Coding Problems (Python/JS).
    *   *Plan:* Add MCQ support with auto-grading in v2.0.

2.  **Full Session Video Recording:**
    *   *Current Status:* We capture snapshots and analytical logs (Risk Scores). We do **not** record the full video stream to save storage.
    *   *Plan:* Add optional cloud recording for high-stakes exams.

3.  **Mobile Support:**
    *   *Current Status:* The UI is responsive, but webcam proctoring is optimized for Desktop/Laptop browsers (Chrome/Firefox).
    *   *Plan:* Native Mobile App integration.

4.  **Advanced Analytics Dashboard:**
    *   *Current Status:* Instructors see simple pass/fail and risk logs.
    *   *Plan:* Add detailed replay timelines and cheat-probability heatmaps.
