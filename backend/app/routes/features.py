from fastapi import APIRouter
from app.services.feature_extraction import extract_features

router = APIRouter()

@router.post("/extract-features")
def features(data: dict):
    frames = data["frames"]
    audio_path = data["audio_path"]

    features = extract_features(frames, audio_path)
    return {"features": features}