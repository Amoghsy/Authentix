# Authentix: Multimodal Deepfake Detection & Explainability Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-19.2-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8.0-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-4.2-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)

**Authentix** is a enterprise-grade, multimodal AI platform designed to detect, analyze, and explain synthetic media manipulations across video visual streams, audio speech tracks, and visual-auditory lip synchronization. By fusing deep learning models with a consensus-driven calibration engine, Authentix protects digital trust and provides human-interpretable forensic reasoning.

---

## 🚨 Problem Statement

The rapid advancement of generative AI (e.g., GANs, diffusion models, voice cloning neural networks) has made creating highly convincing fake videos accessible to anyone. Traditional single-modality deepfake detection tools suffer from major limitations:

1. **Single-Modality Blindspots**: Detectors that focus exclusively on visual artifacts fail when confronted with cloned audio paired with real footage. Conversely, audio-only detectors miss visual face-swaps.
2. **Multi-Vector Attacks**: Modern deepfakes often combine real facial videos with synthetic voiceovers or out-of-sync lip replacements (Lip-Sync attacks).
3. **False Confidence & Modal Contradictions**: Naive score averaging lowers suspicion when one modality is genuine and another is manipulated, causing dangerous false negatives.
4. **Black-Box Output**: Legacy tools produce raw probability scores without explaining *why* a file was flagged, offering no actionable insights for human auditors.

---

## 💡 Solution

Authentix addresses these challenges through a **4-Tiered Multimodal Detection Engine**:

- **Visual Frame Inspection**: MobileNetV3-Small fine-tuned for high-throughput spatial forgery detection.
- **Auditory Speech Analysis**: Pretrained transformer backbones (`Wav2Vec2` / `WavLM`) for synthetic voice and spectral artifact classification.
- **Temporal Lip-Sync Mismatch Engine**: Dual-stream pipeline integrating S3FD face tracking, MediaPipe 478 3D landmarks, and SyncNet audio-visual correlation analysis.
- **Consensus & Threat-Calibrated Fusion Engine**: A multi-modality fusion system ($W_{\text{video}}=0.50$, $W_{\text{audio}}=0.30$, $W_{\text{lipsync}}=0.20$) equipped with variance confidence penalty calibration, single-channel threat escalation, and automated natural language reasoning generation.

---

## ✨ Features

- 🎥 **Multimodal Analysis**: Simultaneously processes visual frames, extracted audio tracks, and temporal lip-to-speech synchronization.
- ⚖️ **Variance-Calibrated Fusion Engine**: Automatically penalizes confidence when modal predictions conflict and escalates risk if any single channel detects a high-risk attack ($>0.70$).
- 📝 **Automated Rule-Based Explainability**: Generates clear, plain-English diagnostic claims explaining exact evidence of tampering or authenticity.
- 📊 **Interactive Analytics Dashboard**: Modern React UI with real-time risk gauges, modal breakdowns, audio playback waveforms, frame previews, and audit logs.
- 📄 **Exportable Forensic PDF Reports**: Automated generation of official downloadable PDF analysis reports containing detailed metrics and diagnostic charts.
- 🔒 **Enterprise Backend Security**: FastAPI REST API equipped with JWT authentication, role-based access control (User/Admin), async database sessions (PostgreSQL/SQLite), and audit logging.
- ⚡ **Optimized Pipeline & Self-Testing**: Modular architecture with automated dependency resolution (FFmpeg, weights auto-download) and CLI self-testing suites.

---

## 🏗️ System Architecture & Workflow

```text
                           ┌────────────────────────┐
                           │      User Upload       │
                           │     (MP4/AVI/MOV)      │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │   FastAPI Web Server   │
                           └───────────┬────────────┘
                                       │
                   ┌───────────────────┼───────────────────┐
                   │                   │                   │
                   ▼                   ▼                   ▼
         ┌───────────────────┐┌───────────────────┐┌───────────────────┐
         │   Video Engine    ││   Audio Engine    ││  Lip-Sync Engine  │
         │ MobileNetV3-Small ││ Wav2Vec2 / WavLM  ││ S3FD + MediaPipe  │
         │ Spatial Forgery   ││ Synthetic Speech  ││    + SyncNet      │
         └─────────┬─────────┘└─────────┬─────────┘└─────────┬─────────┘
                   │                   │                   │
                   └───────────────────┼───────────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │  Multimodal Fusion &   │
                           │   Calibration Engine   │
                           ├────────────────────────┤
                           │ • Weighted Aggregation │
                           │ • Variance Penalty     │
                           │ • Threat Safeguard     │
                           │ • Plain-English Logic  │
                           └───────────┬────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
       ┌───────────────────────┐               ┌───────────────────────┐
       │   Database & Audit    │               │  React Dashboard &    │
       │ (SQLAlchemy / Alembic)│               │  Exportable PDF Report│
       └───────────────────────┘               └───────────────────────┘
```

---

## 🛠️ Tech Stack

### AI / ML Core Engine
- **Frameworks**: PyTorch, Torchaudio, Torchvision, Hugging Face `transformers`
- **Architectures**: MobileNetV3-Small, Wav2Vec2-Base / WavLM, SyncNet (`lithiumice/syncnet`), S3FD Face Detector
- **Computer Vision & Processing**: OpenCV, MediaPipe FaceMesh (478 3D landmarks), Albumentations, Librosa, SoundFile, NumPy, Scikit-Learn

### Backend Architecture
- **Web Framework**: Python 3.11, FastAPI, Uvicorn
- **Database & ORM**: PostgreSQL / SQLite, Async SQLAlchemy v2.0, Alembic Migrations
- **Security & Auth**: PyJWT, Bcrypt, OAuth2 Password Bearer
- **Utilities**: FFmpeg, ReportLab (PDF Generation), Pydantic v2

### Frontend Dashboard
- **Framework & Build**: React 19, Vite 8, React Router v7
- **Styling & UI**: Tailwind CSS v4, Radix UI Primitives, Lucide Icons, Spline 3D Runtime (`@splinetool/react-spline`)

---

## 🔍 Engine-by-Engine Deep Dive

### 1. Video Deepfake Detection Engine 🎥

#### Logic & Pipeline
- **Preprocessing**: Input videos are resampled, frames extracted, and faces detected and normalized to $224 \times 224 \times 3$ RGB tensors.
- **Model Architecture**: Customized `MobileNetV3-Small` backbone initialized with ImageNet weights. The default 1000-class head is replaced with a custom binary classification head (`Linear(576 -> 256) -> Hardswish -> Dropout(0.2) -> Linear(256 -> 2)`).
- **Staged Transfer Learning**:
  - **Phase 1**: Backbone frozen, classifier head trained with AdamW ($LR = 10^{-3}$).
  - **Phase 2**: Unfreeze last 3 Conv/InvertedResidual blocks for localized feature adaptation ($LR = 10^{-4}$).
  - **Phase 3**: Full model fine-tuning with Cosine Annealing Learning Rate Schedule ($LR = 10^{-5}$).

#### Best Evaluation Results
| Metric | Best Value |
| :--- | :--- |
| **ROC-AUC Score** | **86.51%** (`0.8651`) |
| **Validation Accuracy** | **76.80%** |
| **Training Accuracy** | **95.34%** |
| **Inference Throughput** | **~1,832 frames/sec** |

---

### 2. Audio Deepfake Detection Engine 🎙️

#### Logic & Pipeline
- **Preprocessing**: Audio streams are extracted via FFmpeg, converted to $16\text{ kHz}$ mono WAV format, peak-normalized, and sliced into 5-second evaluation windows.
- **Model Architecture**: Pretrained Transformer encoder backbone (`facebook/wav2vec2-base` / `WavLM`) with sequence classification head.
- **Training Strategy**: Two-stage transfer learning fine-tuning. Phase 1 trains only the projection classifier head, while Phase 2 unfreezes the upper N transformer encoder layers to capture fine spectral voice cloning artifacts.

#### Best Evaluation Results
*(Evaluated on clean benchmark test split of 189 audio samples)*
| Metric | Result |
| :--- | :--- |
| **Accuracy** | **100.00%** (`1.0000`) |
| **ROC-AUC** | **1.0000** |
| **Precision** | **1.0000** |
| **Recall (Sensitivity)** | **1.0000** |
| **F1-Score** | **1.0000** |
| **Confusion Matrix** | **TN: 88 (Real), TP: 101 (Fake), FP: 0, FN: 0** |

---

### 3. Lip Sync Mismatch Engine 👄

#### Logic & Pipeline
- **Video Stream Tracking**: Video decoded at 25 FPS. S3FD face detector detects and tracks active speaker bounding boxes across frames.
- **Landmark Mesh & Crop**: MediaPipe FaceMesh locates 478 3D facial landmarks to crop mouth ROIs normalized to $111 \times 111$ grayscale. Crops are batched into 5-frame sliding windows producing PyTorch tensors of shape `(Batch, 1, 5, 111, 111)`.
- **Audio MFCC Extraction**: Audio signal converted into 13-dimensional MFCC features at $100\text{ Hz}$, formatted into 20-frame temporal slices of shape `(Batch, 1, 20, 13)`.
- **SyncNet Correlation**: Evaluated using pretrained `lithiumice/syncnet` model weights. Computes distance metrics between visual lip movements and audio acoustic features to identify temporal synchronization offsets and out-of-sync audio replacements.

---

### 4. Multimodal Fusion & Calibration Engine ⚙️

#### Logic & Mathematical Formulations

1. **Weighted Score Aggregation**:
   $$S_{\text{fused}} = W_{\text{video}} \cdot S_{\text{video}} + W_{\text{audio}} \cdot S_{\text{audio}} + W_{\text{lipsync}} \cdot S_{\text{lipsync}}$$
   *Default Weights*: $W_{\text{video}} = 0.50$, $W_{\text{audio}} = 0.30$, $W_{\text{lipsync}} = 0.20$.

2. **Consensus Variance Confidence Penalty Calibration**:
   When modality predictions disagree (e.g. Video says Fake $0.95$, Audio says Real $0.05$), the system computes weighted variance:
   $$\sigma^2 = \sum_{i} W_i \cdot (S_i - S_{\text{fused}})^2$$
   $$\text{Penalty} = 2.0 \cdot \sigma^2$$
   $$\text{Confidence}_{\text{calibrated}} = \text{Confidence}_{\text{aggregated}} \cdot (1.0 - \text{Penalty})$$
   This penalizes confidence by up to $50\%$ in conflicting scenarios, signaling uncertainty to human operators.

3. **Single-Modality Threat Safeguard**:
   To prevent single-channel attacks (e.g. voice cloning on real video) from being averaged down to "Authentic", if $\max(S_i) > 0.70$:
   $$S_{\text{calibrated}} = \max\left(S_{\text{fused}}, \, 0.40 \cdot S_{\text{fused}} + 0.60 \cdot \max(S_i)\right)$$

4. **Automated Plain-English Explainability**:
   Translates scores, confidence bounds, and modal conflicts into human-readable sentences (e.g., *"Visual manipulation probability is high"*, *"Modality conflict: Audio indicates manipulation while Video appears authentic"*, *"Aggregated deepfake risk is driven by lip-sync synchronization anomalies"*).

---

## 📁 System Modules Overview

```text
Authentix/
├── ai/                         # Deepfake AI Models & Engines
│   ├── audio/                  # Audio Wav2Vec2/WavLM training, inference, testing
│   ├── video/                  # Video MobileNetV3-Small pipeline, trainer, checkpoints
│   ├── lip_sync/               # Lip sync S3FD, MediaPipe 3D Mesh, SyncNet crops
│   ├── fusion/                 # Scoring, variance calibration, reasoning generator
│   └── dataset/                # Dataset loaders and processed artifacts
├── backend/                    # FastAPI Server Core
│   ├── app/
│   │   ├── api/                # API routes (Auth, Analysis, Reports)
│   │   ├── database/           # Async SQLAlchemy ORM models & migrations
│   │   ├── repositories/       # DB Access Layer (Users, Analysis History, Audit)
│   │   └── services/           # Business logic (Video/Audio/Fusion/Report services)
├── frontend/                   # React 19 + Vite Dashboard
│   ├── src/
│   │   ├── components/         # UI components (Risk Dials, Waveform, Upload Zone)
│   │   ├── pages/              # Dashboard, History, Analysis Detail views
│   │   └── services/           # Axios API client & WebSocket handler
└── experiments/                # Model evaluation scripts and benchmark plots
```

---

## 🚀 Quickstart & Setup Guide

### Prerequisites
- **Python 3.11** installed
- **Node.js 18+** and **npm**
- **FFmpeg** installed (automatically downloaded on Windows if missing)

### 1. Environment Setup & Backend Installation
```bash
# Clone the repository
git clone https://github.com/YourRepo/Authentix.git
cd Authentix

# Create and activate Python virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run Database Migrations & Start Backend Server
```bash
# Apply database migrations
alembic upgrade head

# Start FastAPI server
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Backend API Documentation will be live at `http://127.0.0.1:8000/docs`.*

### 3. Frontend Installation & Dev Server
```bash
cd frontend

# Install node dependencies
npm install

# Launch React development server
npm run dev
```
*Frontend App will be running at `http://localhost:5173`.*

---

## 🔮 Future Enhancements

- 🧬 **Frequency-Domain Diffusion Artifact Detection**: Incorporate FFT / DCT spectral analysis to detect subtle generative artifacts from Flux, Sora, and SDXL models.
- ⚡ **Real-Time WebRTC Stream Scanning**: Extend the pipeline to analyze live video feeds during video calls and conferences.
- 🗺️ **Grad-CAM Visual Heatmaps**: Integrate spatial Grad-CAM overlay generation to highlight exact image pixels manipulated in deepfake frames.
- 🌐 **Cross-Lingual Audio Deepfake Support**: Expand fine-tuning on multilingual synthetic audio datasets (Spanish, Hindi, French, Mandarin).
- 🏢 **Enterprise Webhook & Batch Processing**: API endpoints for automated high-volume enterprise media scanning.

---

## 👥 Contributors

Developed with ❤️ by **Team Authentix**:

- **Amogh** — *Lead AI/ML & System Architect*
- **Team Authentix Contributors** — *Full-Stack, Backend & Research Engineers*

---

<p center>
<b>Authentix</b> — Protecting Digital Integrity in the Age of Generative AI.
</p>
