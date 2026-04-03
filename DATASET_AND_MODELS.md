# Detailed Documentation: Datasets and ML Models

This document provides an in-depth explanation of the dataset synthesis, the number of samples used, training methodologies, and a technical breakdown of each Machine Learning engine within the **Temporal Behavioral Inference Engine (TBIE)** framework.

Because true public datasets with continuous temporal trajectories of students cheating do not exist due to rigorous PII (Personally Identifiable Information) laws, the system leverages highly realistic **Autoregressive AR(1)** synthetic generation to synthesize high-stakes exam behavior.

---

## 1. Dataset Generation: The 5,000 Session Paradigm 

### Total Dataset Size
The baseline risk dataset (`generate_noisy_dataset.py`) generates a total of **5,000 independent procedural exam sessions**.
- Every session is represented temporally by **120 consecutive frames** (approx. 4 seconds of human activity).
- Total simulated frames: `5,000 sessions * 120 frames = 600,000 instances` of discrete multi-dimensional vector points.

### The 50/50 Class Split
The dataset models extreme imbalance naturally, but for supervised gradient descent training, it provides a perfectly balanced prior:
- **Class 0 (Genuine / 2,500 Sessions)**: Students exhibiting normal procedural behavior.
- **Class 1 (Anomalous / 2,500 Sessions)**: Simulated cheating events split evenly (625 sessions each) across 4 sophisticated failure modes.

### Analyzing the 4 Anomaly Modalities (Class 1)
These 4 distinct modalities train the specialized sequence models (UC2-UC5):

#### A. Abrupt Impersonation Hand-Off (625 samples)
- **What it simulates:** A student stepping away and a second person immediately taking their seat. 
- **Signal Effect:** Massive, instantaneous drop in UC1 (Identity Similarity), accompanied by spikes in UC2 (Identity Instability).

#### B. Sophisticated Drift (625 samples)
- **What it simulates:** "The Boiling Frog Attack". An adversary slowly shifts camera angle or lighting to bypass simple threshold checks over several minutes. 
- **Signal Effect:** Slow, gradual delta in Identity embedding. Bypasses short-term models like UC2.

#### C. Presence Absence (625 samples)
- **What it simulates:** Leaving the exam desk entirely. 
- **Signal Effect:** Face presence (UC3) completely drops, inducing massive noise in alignment models.

#### D. Flickering Substitution (625 samples)
- **What it simulates:** Utilizing virtual camera injection to rapidly toggle between the student's feed and a pre-recorded loop. 
- **Signal Effect:** Rapid oscillation in the middle-band similarity. Fuses weak identity correlations over sustained 4-second blocks.

---

## 2. Technical Breakdown: What Each Model Does

The TBIE architecture relies on a frozen Computer Vision Backbone passing raw signals into 5 Temporal Experts, culminating in 1 Fusion Engine.

### Backbone: Stage 1 (IDE)
#### ResNet-50 Identity Anchor
*   **What it does:** Extracts a 256-dimensional semantic representation (embedding vector) of a human face from raw RGB pixels.
*   **Mechanism:** Truncated before the specific fully connected classification head, this purely measures the semantic distance between the current frame and the locked "Enrollment Snapshot" taken at the start of the exam.

### Temporal Expert: UC2 (Identify Instability Model)
*   **Architecture:** **LSTM (Long Short-Term Memory)** with 32 hidden dimensions.
*   **Inputs:** Raw cosine similarity sequence ($S_t$) across time.
*   **Dataset:** Extracted from the AR(1) dataset mentioned above.
*   **What it does:** Reacts violently to sudden, unnatural shifts in identity. Real humans shift gracefully. If similarity drops abruptly, UC2 triggers the warning for physical impersonation.

### Temporal Expert: UC3 (Presence and Attentiveness Model / PAM)
*   **Architecture:** **BiDirectional LSTM (BiLSTM)** with 64 hidden dimensions.
*   **Inputs:** 6D Vector (`[Face Confidence, Bounding Box Ratio, Pitch, Yaw, Roll, Optical Flow Momentum]`).
*   **What it does:** Determines if the participant is physically sitting in front of the screen.

### Temporal Expert: UC4 (Long-Term Drift Detector / LDD)
*   **Architecture:** **BiLSTM** with 128 hidden dimensions.
*   **Inputs:** The 257-dimensional displacement vector (The raw delta between today's face embedding and the base embedding).
*   **What it does:** The primary defense against Sophisticated Drift. It views a giant mathematical timeline buffer (120 frames) to detect gradual camera subversions.

### Temporal Expert: GAM (Gaze Attentiveness Model) 
*   **Architecture:** **2-Layer BiLSTM** with 64 hidden dimensions.
*   **Inputs:** 6D Vector (`[Pupil X, Pupil Y, Pitch, Yaw, Blink Rate, Saccade Velocity]`).
*   **Dataset/Training Focus:** Trained heavily on synthesized off-screen target fixation points (reading unseen text on phones or monitors).
*   **What it does:** Assesses purely oculomotor intent. Distinguishes normal "thinking" (staring off randomly) from targeted "reading" behaviors off-camera.

### Temporal Expert: HGDM (Head-Gaze Dynamics Model)
*   **Architecture:** **2-Layer BiLSTM** with 64 hidden dimensions.
*   **Inputs:** 7D Cross-Modal Vector comparing the first derivatives of Head posture $\Delta H$ and Gaze angles $\Delta G$.
*   **What it does:** Evaluates "Kinematic Decoupling." People usually move their head to look at far objects. If the head sits perfectly still but the eyes move wildly to a corner (decorrelation), this signifies probable mobile phone usage beneath the desk. 

### Final Synthesis: UC5 (Risk Fusion Engine / RFE)
*   **Architecture:** **Gated Recurrent Unit (GRU)** with 32 hidden dimensions.
*   **Inputs:** The 6 probability scalar outputs from all temporal models concatenated together `[IDE, UC2, UC3, UC4, GAM, HGDM]`. 
*   **What it does:** It acts as the ultimate "Jury." A single glitch in gaze does not constitute cheating if identity and posture are rock solid. The GRU intelligently weights the competing predictions to output a singular `risk probability` and an `uncertainty bounding` parameter.

---

## 3. Detailed Training Pipeline (Gradient Descent)

### Data Preparation
Because continuous frame analysis suffers from massive class-imbalance, the data is scaled via MinMax Normalization or Z-Score specifically for vectors.

### Training Configuration & Optimization (Across all Recurrent Models)
- **Batch Size:** Generously set (usually $64$ or $128$) to promote stable parameter updates.
- **Optimizer:** Adaptive Moment Estimation (`Adam`).
- **Learning Rate:** Commencing at `0.001` or `0.005`, often utilizing a StepLR or ReduceLROnPlateau scheduler for fine-tuning.
- **Loss Function:** **Binary Cross-Entropy (BCELoss)** tailored for sequential sequence outputs.
- **Epochs:** $20 - 50$ epochs. The BiLSTMs typically converge very rapidly (within 15 epochs) due to the strongly distinct autoregressive signatures built into the dataset's anomalies.
- **Regularization:** Extreme Dropout (`0.3` to `0.5`) injected between LSTM layer stacking to force robust structural representation and avoid rote memorization of simple noise thresholds.

By breaking the problem from single "Images" down into temporal vectors over $T=120$, the network mathematically ignores split-second physical glitches and correctly targets systemic cheating behaviors spanning multiple seconds.
