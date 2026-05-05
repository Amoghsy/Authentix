from fastapi import APIRouter
from app.services.model_service import predict_model

router = APIRouter()

@router.post("/predict-model")
def predict(data: dict):
    """
    Expects: { "features": { "audio": {...}, "video": {...} } }
    Returns modality score breakdown plus fused model_score.
    """
    result = predict_model(data["features"])
    return {**result, "model_used": "DeepFace + RF for video, librosa-derived + LR for audio"}
