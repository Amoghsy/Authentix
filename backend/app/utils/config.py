import os

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths
UPLOAD_DIR = os.path.join(BACKEND_DIR, "data", "uploads")
AUDIO_DIR  = os.path.join(BACKEND_DIR, "data", "audio")

# Processing defaults
FRAME_SIZE       = (224, 224)
MAX_FRAMES       = 15
MIN_FRAMES       = 10
MAX_AUDIO_SECONDS = 10.0

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR,  exist_ok=True)

# ── Decision threshold (binary) ───────────────────────────────────────────
FAKE_THRESHOLD = 0.5   # FAKE >= 0.5 | REAL < 0.5

# ── Video model fusion weights (sum = 1.00) ───────────────────────────────
VIDEO_CNN_WEIGHT      = 0.40
VIDEO_TEMPORAL_WEIGHT = 0.20
VIDEO_FFT_WEIGHT      = 0.15
VIDEO_DEEPFACE_WEIGHT = 0.10
VIDEO_LANDMARK_WEIGHT = 0.10
VIDEO_RF_WEIGHT       = 0.05

# ── Audio model fusion weights (sum = 1.00) ───────────────────────────────
AUDIO_LIBROSA_WEIGHT  = 0.60
AUDIO_LR_WEIGHT       = 0.20
AUDIO_LIPSYNC_WEIGHT  = 0.20

# ── Cross-modal fusion (50 / 50 — audio is equally weighted) ─────────────
VIDEO_MODEL_WEIGHT = 0.50
AUDIO_MODEL_WEIGHT = 0.50

# ── Heuristic layer weights (heuristics kept for logging, not final score) ─
HEURISTIC_VIDEO_WEIGHT = 0.80
HEURISTIC_AUDIO_WEIGHT = 0.20

# ── Final fusion — non-linear + direct signal injection ───────────────────
#
#   Step 1:  model_score = 0.50 * video + 0.50 * audio   (cross-modal above)
#   Step 2:  boosted     = model_score ** FUSION_NONLINEAR_POWER
#   Step 3:  final_score = FUSION_MODEL_WEIGHT  * boosted
#                        + FUSION_LIPSYNC_WEIGHT * lip_sync_score
#                        + FUSION_DEEPFACE_WEIGHT * deepface_score
#
FUSION_NONLINEAR_POWER = 0.85   # x^0.85 — amplifies mid-range scores
FUSION_MODEL_WEIGHT    = 0.65
FUSION_LIPSYNC_WEIGHT  = 0.25   # lip-sync is the strongest fake signal
FUSION_DEEPFACE_WEIGHT = 0.10

# ── Legacy stubs (backward compat — validate.py imports these) ────────────
MODEL_WEIGHT     = FUSION_MODEL_WEIGHT   # 0.65
HEURISTIC_WEIGHT = 0.0                   # heuristics removed from final score

# ── Legacy model artifact constants ──────────────────────────────────────
PRETRAINED_WEIGHT = 0.8
CUSTOM_WEIGHT     = 0.2
