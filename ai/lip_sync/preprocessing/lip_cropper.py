"""
lip_cropper.py

This module contains the LipCropper class, which is responsible for extracting the mouth
region of interest (ROI) from frame images using facial landmarks. It aligns, pads, crops,
converts to grayscale, resizes, normalizes, and saves the sequence of cropped lip images
while preserving temporal ordering.
"""

import logging
from pathlib import Path
import sys
from typing import List, Tuple, Optional
import cv2
import numpy as np

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.lip_sync.preprocessing.config import ProjectConfig, get_default_config


class LipCropError(Exception):
    """Raised when mouth ROI extraction or image operations fail."""
    pass


class LipCropper:
    """
    Crops, resizes, and normalizes mouth regions from frames based on face landmarks,
    ensuring temporal alignment and clean scaling.
    """

    def __init__(self, config: Optional[ProjectConfig] = None):
        """
        Initializes the LipCropper with config parameters.
        """
        self.config = config or get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)

    def crop_mouth(self, frame: np.ndarray, lip_landmarks: np.ndarray) -> np.ndarray:
        """
        Extracts a padded square mouth ROI from a single frame based on landmark coordinates.
        
        Args:
            frame (np.ndarray): BGR frame image.
            lip_landmarks (np.ndarray): NumPy array of shape (N, 3) representing relative coordinates.
            
        Returns:
            np.ndarray: Cropped, normalized mouth ROI (width x height, single channel grayscale or BGR).
            
        Raises:
            LipCropError: If crop bounding box dimensions calculations are invalid.
        """
        h, w, c = frame.shape
        
        # 1. Convert relative landmarks to absolute pixel coordinates
        abs_x = lip_landmarks[:, 0] * w
        abs_y = lip_landmarks[:, 1] * h
        
        # 2. Find bounding extremes
        x_min, x_max = np.min(abs_x), np.max(abs_x)
        y_min, y_max = np.min(abs_y), np.max(abs_y)
        
        mouth_w = x_max - x_min
        mouth_h = y_max - y_min
        
        if mouth_w <= 0 or mouth_h <= 0:
            raise LipCropError("Invalid mouth landmarks. Bounding box coordinates resolve to zero area.")
            
        # 3. Calculate center of the mouth
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        
        # 4. Enforce square bounding box by using the max dimension and applying padding
        padding_factor = self.config.lip.padding
        box_size = max(mouth_w, mouth_h) * (1.0 + 2.0 * padding_factor)
        
        # 5. Calculate crop boundaries
        x1 = int(round(cx - box_size / 2.0))
        y1 = int(round(cy - box_size / 2.0))
        x2 = int(round(cx + box_size / 2.0))
        y2 = int(round(cy + box_size / 2.0))
        
        # 6. Extract region, padding with black pixels if coordinates fall out of frame boundary
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - w)
        pad_bottom = max(0, y2 - h)
        
        # Clip boundaries to image edges
        x1_clamped = max(0, x1)
        y1_clamped = max(0, y1)
        x2_clamped = min(w, x2)
        y2_clamped = min(h, y2)
        
        crop = frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
        
        # Apply padding if necessary to maintain square format and center the mouth
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(
                crop,
                top=pad_top,
                bottom=pad_bottom,
                left=pad_left,
                right=pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )
            
        # 7. Convert to Grayscale if configured (SyncNet default)
        if self.config.lip.grayscale:
            if len(crop.shape) == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                
        # 8. Resize to target dimension
        target_size = self.config.lip.crop_size
        crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        
        # 9. Peak Normalization (scale pixel values to float [0.0, 1.0])
        crop_normalized = crop_resized.astype(np.float32) / 255.0
        
        return crop_normalized

    def crop_sequence(
        self,
        frames: List[np.ndarray],
        landmarks_sequence: List[Optional[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Extracts mouth ROIs for a temporal list of frames.
        Uses adjacent bounding boxes as a fallback if landmarks are missing in isolated frames.
        
        Args:
            frames (List[np.ndarray]): Input frames list.
            landmarks_sequence (List[Optional[np.ndarray]]): Matching face landmarks list.
            
        Returns:
            List[np.ndarray]: List of cropped, normalized mouth matrices in identical order.
        """
        if len(frames) != len(landmarks_sequence):
            raise LipCropError(
                f"Frames count ({len(frames)}) does not match landmark sequence count ({len(landmarks_sequence)})."
            )
            
        crops: List[np.ndarray] = []
        last_valid_landmarks: Optional[np.ndarray] = None
        
        # Scan forward first to find the first valid landmark frame
        for lm in landmarks_sequence:
            if lm is not None:
                last_valid_landmarks = lm
                break
                
        if last_valid_landmarks is None:
            raise LipCropError("No valid landmarks found in the entire sequence. Cannot perform lip cropping.")
            
        for i, (frame, lm) in enumerate(zip(frames, landmarks_sequence)):
            if lm is not None:
                last_valid_landmarks = lm
            
            # Crop using the current or closest valid temporal landmarks
            try:
                crop = self.crop_mouth(frame, last_valid_landmarks)
                crops.append(crop)
            except Exception as e:
                self.logger.error(f"Error cropping mouth for frame index {i}: {e}")
                # Fallback: append a blank frame or copy the previous crop
                if len(crops) > 0:
                    crops.append(crops[-1].copy())
                else:
                    target_size = self.config.lip.crop_size
                    crops.append(np.zeros((target_size, target_size), dtype=np.float32))
                    
        return crops

    def save_crops(self, crops: List[np.ndarray], output_dir: Path, prefix: str = "mouth") -> List[Path]:
        """
        Saves cropped lip frames to disk in temporal order.
        Converts float pixel matrices back to uint8 [0, 255] for standard file write.
        
        Args:
            crops (List[np.ndarray]): List of normalized mouth float arrays.
            output_dir (Path): Destination folder.
            prefix (str): Filename output prefix.
            
        Returns:
            List[Path]: List of absolute paths to saved files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []
        
        for idx, crop in enumerate(crops):
            filename = f"{prefix}_{idx:05d}.jpg"
            file_path = output_dir / filename
            
            # Convert back from float32 [0.0, 1.0] to uint8 [0, 255]
            img_to_save = (crop * 255.0).astype(np.uint8)
            
            success = cv2.imwrite(str(file_path), img_to_save)
            if not success:
                self.logger.error(f"Failed to write crop image to: {file_path}")
                raise LipCropError(f"Failed to save crop image file at: {file_path}")
                
            saved_paths.append(file_path)
            
        self.logger.info(f"Successfully saved {len(crops)} mouth crop frames to: {output_dir}")
        return saved_paths


if __name__ == "__main__":
    print("Executing self-test for lip_cropper.py...")
    import tempfile
    
    cropper = LipCropper()
    
    # Generate mock frame and lip landmarks centered around 0.5
    height, width = 200, 200
    mock_frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw a white box in the center representing mouth area
    mock_frame[80:120, 80:120] = 255
    
    # 40 mock landmarks representing a square mouth (relative coordinates)
    # Bounding extremes: x_min=0.4, x_max=0.6, y_min=0.4, y_max=0.6
    landmarks = []
    for x in [0.4, 0.6]:
        for y in np.linspace(0.4, 0.6, 20):
            landmarks.append([x, y, 0.0])
    mock_landmarks = np.array(landmarks)
    
    try:
        # Test individual crop
        crop = cropper.crop_mouth(mock_frame, mock_landmarks)
        print(f"Cropped ROI shape: {crop.shape}, max_val={np.max(crop):.2f}, dtype={crop.dtype}")
        
        # Verify shape (default 111x111) and normalized range
        assert crop.shape == (111, 111)
        assert crop.dtype == np.float32
        assert np.max(crop) <= 1.0
        
        # Test sequence crop
        crops = cropper.crop_sequence([mock_frame, mock_frame], [mock_landmarks, None])
        print(f"Processed sequence crop. Frames count: {len(crops)}")
        assert len(crops) == 2
        assert crops[1].shape == (111, 111)
        
        # Test Save
        temp_dir = Path(tempfile.mkdtemp())
        paths = cropper.save_crops(crops, temp_dir)
        print(f"Saved crops files: {[p.name for p in paths]}")
        assert len(paths) == 2
        assert paths[0].exists()
        
        # Cleanup
        for p in paths:
            if p.exists():
                p.unlink()
        temp_dir.rmdir()
        
        print("All lip_cropper.py self-tests: PASSED")
        
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
