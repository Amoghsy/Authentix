from fastapi import APIRouter, UploadFile, File
import uuid, os

router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-video")
async def upload_video(video_file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    file_path = f"{UPLOAD_DIR}/{video_id}.mp4"

    with open(file_path, "wb") as f:
        f.write(await video_file.read())

    return {"video_id": video_id, "path": file_path}