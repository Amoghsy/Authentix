"""
video_processing.py
───────────────────
Frame extraction and face detection pipeline for Authentix.

Upgraded:
    • Extract 10–15 frames (was 5)
    • Face detection via DeepFace (RetinaFace backend)
    • Face crop before downstream processing
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from app.utils.config import FRAME_SIZE, MAX_FRAMES, MIN_FRAMES

logger = logging.getLogger(__name__)


def extract_frames(video_path: str) -> List[np.ndarray]:
    """
    Extract up to MAX_FRAMES frames from the video, evenly distributed.
    Returns a list of numpy uint8 arrays of shape (224, 224, 3).
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Determine how many frames to grab — at least MIN_FRAMES, up to MAX_FRAMES
    target_frames = min(MAX_FRAMES, max(MIN_FRAMES, int(duration_sec)))
    # Calculate frame interval for even distribution
    interval = max(1, total_frames // target_frames) if total_frames > 0 else fps

    frames = []
    try:
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % interval == 0:
                frame = cv2.resize(frame, FRAME_SIZE)  # shape: (224,224,3) uint8
                frames.append(frame)
                if len(frames) >= MAX_FRAMES:
                    break

            count += 1
    finally:
        cap.release()

    logger.info("Extracted %d frames from %s (interval=%d, total=%d, fps=%d)",
                len(frames), video_path, interval, total_frames, fps)
    return frames


def detect_and_crop_faces(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Detect faces in frames using DeepFace (RetinaFace backend) and return
    both cropped face regions and the original frames.

    Returns:
        (face_crops, original_frames)
        - face_crops: list of cropped face images resized to FRAME_SIZE
        - original_frames: the input frames (passed through unchanged)

    If face detection fails for a frame, the full frame is used as fallback.
    """
    try:
        from deepface import DeepFace  # noqa: PLC0415
    except ImportError:
        logger.warning("DeepFace not available for face detection — using full frames")
        return frames, frames

    face_crops = []
    for idx, frame in enumerate(frames):
        try:
            faces = DeepFace.extract_faces(
                frame,
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
            )
            if faces and len(faces) > 0:
                # Use the first (largest) detected face
                face_region = faces[0].get("face", None)
                if face_region is not None:
                    # DeepFace returns face as float [0,1], convert to uint8
                    if face_region.max() <= 1.0:
                        face_region = (face_region * 255).astype(np.uint8)
                    face_resized = cv2.resize(face_region, FRAME_SIZE)
                    face_crops.append(face_resized)
                    continue
        except Exception as exc:
            logger.debug("Face detection failed for frame %d: %s", idx, exc)

        # Fallback: use full frame
        face_crops.append(frame)

    logger.info("Face detection complete: %d/%d faces cropped", len(face_crops), len(frames))
    return face_crops, frames