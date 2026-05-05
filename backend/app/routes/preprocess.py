from fastapi import APIRouter
from app.services.video_processing import extract_frames
from app.services.audio_processing import extract_audio

router = APIRouter()

@router.post("/preprocess")
def preprocess(data: dict):
    video_path = data["path"]

    frames = extract_frames(video_path)
    audio_path = extract_audio(video_path)

    return {"frames": frames, "audio_path": audio_path}