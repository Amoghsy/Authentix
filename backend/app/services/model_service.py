"""
model_service.py
────────────────
Thin service-layer adapter that bridges the FastAPI route layer and the
model_loader inference engine.

Responsibilities:
    1. Trigger model loading once at import time (server start-up).
    2. Expose predict_model() as the single callable for the pipeline.
"""

from app.models.model_loader import load_models, predict

# Load models once at module import time (triggered on server start-up).
load_models()


def predict_model(features: dict) -> dict:
    """
    Run the hybrid model inference pipeline.

    Args:
        features: Dict produced by feature_extraction.extract_features():
            {
                "video":          { variance, flicker, blur },
                "audio":          { mfcc_mean, pitch_var, energy },
                "frames":         List[np.ndarray],
                "face_crops":     List[np.ndarray],
                "audio_path":     str,
                "video_available": bool,
                "audio_available": bool,
            }

    Returns:
        {
            "cnn_score":         float,
            "deepface_score":    float,
            "temporal_score":    float,
            "fft_score":         float,
            "landmark_score":    float,
            "texture_score":     float,
            "rf_score":          float,
            "video_model_score": float,
            "librosa_score":     float,
            "lr_score":          float,
            "lip_sync_score":    float,
            "audio_model_score": float,
            "model_score":       float,
        }
    """
    return predict(features)
