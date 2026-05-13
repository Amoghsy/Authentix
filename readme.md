# 🔐 Authentix — AI-Powered Deepfake Detection Platform

Authentix is a full-stack AI-powered deepfake detection system that analyzes uploaded videos using multimodal machine learning models. The platform combines **computer vision**, **audio analysis**, **temporal consistency checks**, and **hybrid AI fusion techniques** to determine whether a video is **REAL** or **FAKE** with explainable insights.

---

# 🚀 Features

* 🎥 Upload and analyze `.mp4` videos
* 🧠 9 AI -model multimodal AI inference pipeline
* 👁️ Face detection and visual artifact analysis
* 🔊 Audio extraction and lip-sync verification
* 📊 Confidence scores with detailed breakdown
* 📖 Human-readable AI explanations
* ⚡ FastAPI backend with React frontend
* 🎨 Modern responsive UI
* 🔍 Explainable AI decision system

---

# 🏗️ Project Architecture

```text
Authentix/
│
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── services/        # AI detection services
│   │   ├── routes/          # API routes
│   │   ├── utils/           # Helper utilities
│   │   └── main.py          # FastAPI entry point
│   │
│   └── requirements.txt
│
├── frontend/                # React Frontend
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── services/
│   │   └── App.jsx
│   │
│   └── package.json
│
└── README.md
```

---

# 🌐 Frontend Workflow

## 1. Landing Page (`/`)

* User visits the Authentix homepage
* Introduces the platform and features

## 2. Video Upload (`/detect`)

* User uploads a `.mp4` file
* Frontend sends the video to backend API

## 3. Processing State

* UI displays loading animations
* Backend runs AI analysis

## 4. Results Page (`/analysis`)

Displays:

* ✅ Final Verdict (`REAL` / `FAKE`)
* 📈 Confidence Score
* 🧠 Individual Model Scores
* 📖 AI-generated Explanation

---

# ⚙️ Backend Deepfake Detection Pipeline

## 1. File Ingestion

* Receives uploaded video
* Generates UUID
* Stores video temporarily

---

## 2. Preprocessing Engine

### 🎥 Video Processing (OpenCV)

* Extract frames
* Detect faces
* Crop facial regions

### 🔊 Audio Processing (FFmpeg)

* Extract audio as `.wav`
* Handle no-audio fallback

---

## 3. Feature Extraction

### Visual Features

* Frame variance
* Blur detection
* Lighting flicker analysis

### Audio Features

* MFCC extraction
* Pitch variance
* Energy levels

---

## 4. Multimodal AI Inference

### 🎬 Video Models

| Model             | Purpose                     |
| ----------------- | --------------------------- |
| CNN               | Spatial artifact detection  |
| Temporal Model    | Motion consistency          |
| FFT               | Frequency artifact analysis |
| DeepFace          | Facial inconsistency        |
| Landmark Detector | Facial structure analysis   |
| Random Forest     | Ensemble classification     |

### 🔊 Audio Models

| Model               | Purpose                     |
| ------------------- | --------------------------- |
| Librosa             | Audio anomaly analysis      |
| Logistic Regression | Audio classification        |
| Lip-Sync Model      | Audio-video synchronization |

---

## 5. Heuristic Analysis

Rule-based checks for:

* Unnatural blur
* Abnormal audio energy
* Temporal inconsistencies

---

## 6. Hybrid Fusion Layer

The final decision combines:

* Video model scores
* Audio model scores
* DeepFace weighting
* Lip-sync weighting
* Non-linear uncertainty boosting

Final score is normalized between:

```python
0.0 → REAL
1.0 → FAKE
```

---

## 7. Decision Engine

```python
if score >= 0.5:
    verdict = "FAKE"
else:
    verdict = "REAL"
```

---

## 8. Explainability Layer

Generates:

* Human-readable explanations
* Dominant contributing signals
* Model influence summaries

Example:

```text
"The video is likely FAKE due to strong lip-sync mismatch,
facial inconsistencies detected by DeepFace,
and temporal artifacts across frames."
```

---

# 🧪 Tech Stack

## Frontend

* React.js
* Axios
* Tailwind CSS
* Framer Motion

## Backend

* FastAPI
* Python
* OpenCV
* FFmpeg
* Librosa
* Scikit-learn
* DeepFace
* NumPy

---

# 📡 API Endpoint

## Detect Deepfake

```http
POST /detect
```

### Request

```multipart/form-data
video: file.mp4
```

### Response

```json
{
  "verdict": "FAKE",
  "confidence": 87.4,
  "signals": {
    "cnn": 0.89,
    "temporal": 0.82,
    "deepface": 0.93,
    "lip_sync": 0.92
  },
  "explanation": "The video is likely fake due to lip-sync mismatch..."
}
```

---

# 🛠️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/authentix.git
cd authentix
```

---

# 🔧 Backend Setup

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Backend

```bash
uvicorn app.main:app --reload
```

Backend runs on:

```text
http://localhost:8000
```

---

# 🎨 Frontend Setup

## Install Dependencies

```bash
npm install
```

---

## Run Frontend

```bash
npm run dev
```

Frontend runs on:

```text
http://localhost:5173
```

---

# 📊 Workflow Diagram

```text
Frontend Upload
       ↓
FastAPI Backend
       ↓
Video + Audio Preprocessing
       ↓
Feature Extraction
       ↓
9 AI Models
       ↓
Hybrid Fusion Layer
       ↓
Verdict + Explanation
       ↓
Frontend Results UI
```

---

# 🔒 Future Improvements

* 🧠 Transformer-based video analysis
* ☁️ Cloud deployment support
* 📱 Mobile application
* 🎯 Real-time webcam detection
* 🌍 Multi-language explainability
* 📈 Analytics dashboard
* 🔥 GPU acceleration

---

# 🤝 Contributing

Contributions are welcome.

```bash
Fork → Clone → Create Branch → Commit → Push → Pull Request
```

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Authors

Developed with ❤️ by the Authentix Team.

---

# ⭐ Support

If you like this project:

* ⭐ Star the repository
* 🍴 Fork the project
* 🧠 Contribute improvements
* 📢 Share with others
