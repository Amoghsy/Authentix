"""
landmark_detector.py

This module contains the LandmarkDetector class, which leverages MediaPipe's FaceLandmarker
Tasks API (ver 1.0.0+) to extract 468 3D facial landmarks from image frames. It identifies
and extracts lip landmarks using the FACE_LANDMARKS_LIPS connections, and supports localized
face bounding box cropping to maximize detection accuracy.

If the face_landmarker.task model file is missing, the module automatically downloads it
from Google's official storage repository to the checkpoints directory.
"""

import logging
from pathlib import Path
import sys
import urllib.request
from typing import List, Tuple, Optional
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm

from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.lip_sync.preprocessing.config import ProjectConfig, get_default_config


class LandmarkDetectionError(Exception):
    """Raised when facial landmark detection fails or model initialization fails."""
    pass


class LandmarkDetector:
    """
    Integrates MediaPipe FaceLandmarker (Tasks API) to locate 468 facial landmarks
    and isolate mouth regions.
    """

    MODEL_DOWNLOAD_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    def __init__(self, config: Optional[ProjectConfig] = None):
        """
        Initializes the LandmarkDetector, loads the model, and resolves lip indices.
        """
        self.config = config or get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.model_path = self.config.paths.checkpoints_dir / "face_landmarker.task"
        
        # 1. Resolve and download face landmarker task file if missing
        self._ensure_model_file()
        
        # 2. Initialize FaceLandmarker
        try:
            base_options = BaseOptions(model_asset_path=str(self.model_path))
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionTaskRunningMode.IMAGE,
                num_faces=1
            )
            self.landmarker = FaceLandmarker.create_from_options(options)
            
            # 3. Resolve lip indices dynamically
            self._resolve_lip_indices()
            self.logger.info(f"Initialized LandmarkDetector with {len(self.lip_indices)} lip indices.")
            
        except Exception as e:
            raise LandmarkDetectionError(f"Failed to initialize FaceLandmarker: {e}")

    def _ensure_model_file(self) -> None:
        """Downloads the face_landmarker.task model file if not locally cached."""
        if self.model_path.exists():
            self.logger.info(f"Loaded existing face landmarker model from: {self.model_path}")
            return
            
        self.config.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logger.warning(f"Face landmarker model task file not found. Downloading: {self.MODEL_DOWNLOAD_URL}")
        
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="face_landmarker.task") as t:
                urllib.request.urlretrieve(
                    self.MODEL_DOWNLOAD_URL,
                    filename=str(self.model_path),
                    reporthook=t.update_to
                )
            self.logger.info("Face landmarker model downloaded successfully.")
        except Exception as e:
            if self.model_path.exists():
                self.model_path.unlink()
            raise LandmarkDetectionError(f"Failed to download FaceLandmarker model. Error: {e}")

    def _resolve_lip_indices(self) -> None:
        """
        Collects all unique landmark indices associated with the lips from FaceLandmarksConnections.
        """
        lip_set = set()
        for connection in FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
            lip_set.add(connection.start)
            lip_set.add(connection.end)
            
        self.lip_indices = sorted(list(lip_set))
        
        if len(self.lip_indices) == 0:
            self.logger.warning("Dynamic lip indices resolve failed. Using default fallback map.")
            # Fallback map for 468 standard facemesh lip coordinates
            self.lip_indices = [
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95
            ]
            self.lip_indices = sorted(list(set(self.lip_indices)))

    def detect_landmarks(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Detects 468 face landmarks. If a face bounding box is provided, performs detection
        on the cropped face region to maximize coordinates resolution, then maps values back.
        
        Args:
            frame (np.ndarray): BGR frame image.
            bbox (Optional[Tuple[int, int, int, int]]): Face box (x1, y1, x2, y2).
            
        Returns:
            Optional[np.ndarray]: Array of shape (468, 3) storing normalized (x, y, z) coords,
            or None if no face is detected.
        """
        h, w, c = frame.shape
        
        # Crop to face bounding box if provided to maximize landmark resolution
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            if crop_w > 0 and crop_h > 0:
                face_crop = frame[y1:y2, x1:x2]
                rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                
                result = self.landmarker.detect(mp_crop)
                if result.face_landmarks:
                    mesh = result.face_landmarks[0]
                    coords = np.zeros((468, 3), dtype=np.float32)
                    for idx, lm in enumerate(mesh):
                        if idx >= 468:
                            break
                        # Map relative crop coordinates back to absolute frame coordinate space
                        abs_x = (x1 + lm.x * crop_w) / w
                        abs_y = (y1 + lm.y * crop_h) / h
                        abs_z = (lm.z * crop_w) / w
                        coords[idx] = [abs_x, abs_y, abs_z]
                    return coords
                    
        # Fallback / Default: Process entire frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
            
        mesh = result.face_landmarks[0]
        coords = np.zeros((468, 3), dtype=np.float32)
        for idx, lm in enumerate(mesh):
            if idx >= 468:
                break
            coords[idx] = [lm.x, lm.y, lm.z]
            
        return coords

    def get_lip_landmarks(self, face_landmarks: np.ndarray) -> np.ndarray:
        """
        Extracts only the landmarks associated with the lips.
        
        Args:
            face_landmarks (np.ndarray): Full face landmarks array of shape (468, 3).
            
        Returns:
            np.ndarray: Isolated lip coordinates array.
        """
        return face_landmarks[self.lip_indices]

    def get_lip_indices(self) -> List[int]:
        """
        Returns the sorted index positions of mouth landmarks.
        """
        return self.lip_indices


if __name__ == "__main__":
    print("Executing self-test for landmark_detector.py...")
    
    try:
        detector = LandmarkDetector()
        print(f"Dynamically mapped lip indices: {detector.get_lip_indices()[:10]}... (Total={len(detector.get_lip_indices())})")
        assert len(detector.get_lip_indices()) > 0
    except Exception as e:
        print(f"Failed initialization: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Test blank frame case (should return None safely)
    blank_img = np.zeros((320, 240, 3), dtype=np.uint8)
    coords = detector.detect_landmarks(blank_img)
    print(f"Blank image landmark detection result: {coords}")
    assert coords is None
    
    # Generate mock coordinates and slice lip subset
    mock_mesh = np.arange(468 * 3).reshape(468, 3)
    lip_subset = detector.get_lip_landmarks(mock_mesh)
    print(f"Mock lip landmarks shape: {lip_subset.shape}")
    assert lip_subset.shape == (len(detector.get_lip_indices()), 3)
    
    print("All landmark_detector.py self-tests: PASSED")
