import os
import sys
import numpy as np
import joblib
import librosa
import cv2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# =============================================================================
# PATH CONFIGURATION
# All paths resolved relative to THIS FILE so the script works regardless of
# which directory you run it from (fixes the "Loaded 0 samples" bug when
# running from the wrong CWD).
# =============================================================================
THIS_FILE   = os.path.abspath(__file__)                    # .../backend/app/models/train.py
MODEL_DIR   = os.path.dirname(THIS_FILE)                   # .../backend/app/models/
APP_DIR     = os.path.dirname(MODEL_DIR)                   # .../backend/app/
BACKEND_DIR = os.path.dirname(APP_DIR)                     # .../backend/
# Data lives inside backend/ (not at the project root)
BASE_DIR    = BACKEND_DIR                                  # .../backend/

AUDIO_REAL = os.path.join(BASE_DIR, "data", "audio", "real")
AUDIO_FAKE = os.path.join(BASE_DIR, "data", "audio", "fake")
VIDEO_REAL = os.path.join(BASE_DIR, "data", "video", "real")
VIDEO_FAKE = os.path.join(BASE_DIR, "data", "video", "fake")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

MIN_SAMPLES_PER_CLASS = 2   # need at least this many per class to train + split


def _check_dir(path: str, label: str) -> bool:
    """Return True if directory exists and has supported media files."""
    if not os.path.isdir(path):
        print(f"  [ERROR] Directory not found: {path}")
        return False
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    if len(files) == 0:
        print(f"  [ERROR] Directory is empty: {path}")
        return False
    print(f"  [OK]    {label}: {path}  ({len(files)} files)")
    return True


# =============================================================================
# AUDIO FEATURE EXTRACTION
# Features: [mfcc_mean, pitch_var, energy]  — 3 features, must match inference
# =============================================================================
def extract_audio_features(path: str):
    """
    Extract 3 audio features from a .wav file.
    Returns a list of 3 floats, or None if extraction fails.
    """
    try:
        audio, sr = librosa.load(path, sr=None, duration=10.0)

        if len(audio) == 0:
            return None

        mfcc   = float(np.mean(librosa.feature.mfcc(y=audio, sr=sr)))
        pitch  = float(np.var(librosa.yin(audio, fmin=50, fmax=300)))
        energy = float(np.mean(librosa.feature.rms(y=audio)))

        return [mfcc, pitch, energy]

    except Exception as e:
        print(f"    [WARN] Audio failed ({os.path.basename(path)}): {e}")
        return None


# =============================================================================
# VIDEO FEATURE EXTRACTION
# Features: [variance, flicker, blur]  — 3 features, must match inference
# =============================================================================
def extract_video_features(path: str):
    """
    Extract 3 video features from up to 5 frames of a video.
    Returns a list of 3 floats, or None if extraction fails.
    """
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

        frames = []
        while len(frames) < 5:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return None

        arr      = np.array(frames, dtype=np.float32)
        variance = float(np.var(arr))
        flicker  = float(np.mean(np.abs(np.diff(arr, axis=0)))) if len(arr) > 1 else 0.0
        blur     = float(np.mean([cv2.Laplacian(f, cv2.CV_64F).var() for f in frames]))

        return [variance, flicker, blur]

    except Exception as e:
        print(f"    [WARN] Video failed ({os.path.basename(path)}): {e}")
        return None


# =============================================================================
# DATA LOADERS
# =============================================================================
def load_audio_data():
    X, y = [], []
    for label, folder in [(0, AUDIO_REAL), (1, AUDIO_FAKE)]:
        tag = "REAL" if label == 0 else "FAKE"
        files = [f for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in AUDIO_EXTS and not f.startswith(".")]
        ok = 0
        for file in files:
            feats = extract_audio_features(os.path.join(folder, file))
            if feats is not None:
                X.append(feats)
                y.append(label)
                ok += 1
        print(f"    Audio {tag}: {ok}/{len(files)} files loaded")

    return np.array(X, dtype=np.float64), np.array(y, dtype=int)


def load_video_data():
    X, y = [], []
    for label, folder in [(0, VIDEO_REAL), (1, VIDEO_FAKE)]:
        tag = "REAL" if label == 0 else "FAKE"
        files = [f for f in os.listdir(folder)
                 if os.path.splitext(f)[1].lower() in VIDEO_EXTS and not f.startswith(".")]
        ok = 0
        for file in files:
            feats = extract_video_features(os.path.join(folder, file))
            if feats is not None:
                X.append(feats)
                y.append(label)
                ok += 1
        print(f"    Video {tag}: {ok}/{len(files)} files loaded")

    return np.array(X, dtype=np.float64), np.array(y, dtype=int)


# =============================================================================
# TRAINING
# =============================================================================
def train_models():
    print("=" * 60)
    print("  Authentix – Deepfake Detection Model Trainer")
    print("=" * 60)

    # --- Verify data directories exist before loading ---
    print("\n[Step 1] Verifying data directories...")
    dirs_ok = all([
        _check_dir(AUDIO_REAL, "Audio REAL"),
        _check_dir(AUDIO_FAKE, "Audio FAKE"),
        _check_dir(VIDEO_REAL, "Video REAL"),
        _check_dir(VIDEO_FAKE, "Video FAKE"),
    ])
    if not dirs_ok:
        print("\n[FATAL] One or more data directories are missing or empty.")
        print(f"  Expected structure under: {BASE_DIR}")
        print("    data/audio/real/  <- real speech .wav files")
        print("    data/audio/fake/  <- deepfake audio .wav files")
        print("    data/video/real/  <- real face video .mp4 files")
        print("    data/video/fake/  <- deepfake video .mp4 files")
        sys.exit(1)

    # =========================================================
    # AUDIO MODEL  (Logistic Regression)
    # =========================================================
    print("\n[Step 2] Loading audio data...")
    X_audio, y_audio = load_audio_data()
    n_real_a = int(np.sum(y_audio == 0))
    n_fake_a = int(np.sum(y_audio == 1))
    print(f"  Total audio samples: {len(X_audio)}  (REAL={n_real_a}, FAKE={n_fake_a})")

    if len(X_audio) < MIN_SAMPLES_PER_CLASS * 2:
        print(f"  [FATAL] Need at least {MIN_SAMPLES_PER_CLASS * 2} audio samples to train.")
        sys.exit(1)

    print("\n[Step 3] Training audio model (Logistic Regression)...")
    scaler_audio = StandardScaler()
    X_audio_scaled = scaler_audio.fit_transform(X_audio)

    # Use stratify only when both classes have enough samples for the split
    try:
        X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
            X_audio_scaled, y_audio, test_size=0.2, random_state=42, stratify=y_audio
        )
    except ValueError:
        # Not enough samples per class for stratified split – fall back
        X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
            X_audio_scaled, y_audio, test_size=0.2, random_state=42
        )

    audio_model = LogisticRegression(max_iter=1000, C=1.0)
    audio_model.fit(X_tr_a, y_tr_a)
    audio_acc = accuracy_score(y_te_a, audio_model.predict(X_te_a))
    print(f"  Audio model accuracy: {audio_acc:.2%}")
    print(classification_report(y_te_a, audio_model.predict(X_te_a),
                                target_names=["REAL", "FAKE"], zero_division=0))

    # =========================================================
    # VIDEO MODEL  (Random Forest)
    # =========================================================
    print("\n[Step 4] Loading video data...")
    X_video, y_video = load_video_data()
    n_real_v = int(np.sum(y_video == 0))
    n_fake_v = int(np.sum(y_video == 1))
    print(f"  Total video samples: {len(X_video)}  (REAL={n_real_v}, FAKE={n_fake_v})")

    if len(X_video) < MIN_SAMPLES_PER_CLASS * 2:
        print(f"  [FATAL] Need at least {MIN_SAMPLES_PER_CLASS * 2} video samples to train.")
        sys.exit(1)

    print("\n[Step 5] Training video model (Random Forest)...")
    scaler_video = StandardScaler()
    X_video_scaled = scaler_video.fit_transform(X_video)

    try:
        X_tr_v, X_te_v, y_tr_v, y_te_v = train_test_split(
            X_video_scaled, y_video, test_size=0.2, random_state=42, stratify=y_video
        )
    except ValueError:
        X_tr_v, X_te_v, y_tr_v, y_te_v = train_test_split(
            X_video_scaled, y_video, test_size=0.2, random_state=42
        )

    video_model = RandomForestClassifier(n_estimators=100, random_state=42)
    video_model.fit(X_tr_v, y_tr_v)
    video_acc = accuracy_score(y_te_v, video_model.predict(X_te_v))
    print(f"  Video model accuracy: {video_acc:.2%}")
    print(classification_report(y_te_v, video_model.predict(X_te_v),
                                target_names=["REAL", "FAKE"], zero_division=0))

    # =========================================================
    # SAVE ALL FOUR ARTEFACTS
    # =========================================================
    print("\n[Step 6] Saving models and scalers...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(audio_model,  os.path.join(MODEL_DIR, "audio_model.pkl"))
    joblib.dump(video_model,  os.path.join(MODEL_DIR, "video_model.pkl"))
    joblib.dump(scaler_audio, os.path.join(MODEL_DIR, "scaler_audio.pkl"))
    joblib.dump(scaler_video, os.path.join(MODEL_DIR, "scaler_video.pkl"))

    for name in ["audio_model.pkl", "video_model.pkl", "scaler_audio.pkl", "scaler_video.pkl"]:
        path = os.path.join(MODEL_DIR, name)
        size = os.path.getsize(path)
        print(f"  Saved: {name}  ({size:,} bytes)")

    print("\n" + "=" * 60)
    print(f"  Audio accuracy : {audio_acc:.2%}")
    print(f"  Video accuracy : {video_acc:.2%}")
    print(f"  Models saved to: {MODEL_DIR}")
    print("  Training complete!")
    print("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    train_models()