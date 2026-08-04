"""
face_detector.py

This module provides face detection, cropping, and resizing capabilities using the
MediaPipe Tasks API. It identifies the largest face in each frame, crops it with a
configurable padding, forces a square bounding box to prevent aspect ratio distortion,
scales the crop to 224x224 pixels, and saves the resulting image. It supports batch
processing of video frames and logs any failures or frames without faces.
"""

from datetime import datetime
import json
import logging
import multiprocessing
import sys
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple, Optional
import urllib.request

import cv2
import numpy as np
from tqdm import tqdm

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetector:
    """
    Wrapper around MediaPipe Tasks Face Detection to process images, detect faces,
    and generate square cropped face outputs.
    """
    def __init__(self, config: AppConfig) -> None:
        """
        Initializes the MediaPipe face detector, downloading the task model if missing.
        
        Args:
            config (AppConfig): The application configuration containing project root.
        """
        # Ensure models directory exists
        model_dir = config.paths.project_root / "ai" / "video" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "blaze_face_short_range.tflite"
        
        # Download model if not present
        if not model_path.exists():
            logging.info(f"MediaPipe face detection model not found. Downloading to {model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            try:
                urllib.request.urlretrieve(url, str(model_path))
                logging.info("Model download completed successfully.")
            except Exception as e:
                logging.critical(f"Failed to download MediaPipe face detection model: {e}")
                raise e
                
        # Configure MediaPipe Face Detector Options
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=config.pipeline.mp_detection_confidence
        )
        
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect_largest_face(
        self, 
        image_bgr: np.ndarray, 
        min_face_size: int = 50
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects all faces in the image and returns the bounding box of the largest face.
        
        Args:
            image_bgr (np.ndarray): The image in BGR format.
            min_face_size (int): Minimum width or height in pixels to accept.
            
        Returns:
            Optional[Tuple[int, int, int, int]]: (xmin, ymin, width, height) of the largest face,
                                                 or None if no face is detected or face is too small.
        """
        # Convert BGR (OpenCV format) to RGB (expected by MediaPipe Tasks)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Run detection
        results = self.detector.detect(mp_image)
        
        if not results or not results.detections:
            return None
            
        largest_face: Optional[Tuple[int, int, int, int]] = None
        largest_area = -1
        
        for detection in results.detections:
            box = detection.bounding_box
            xmin = box.origin_x
            ymin = box.origin_y
            w = box.width
            h = box.height
            
            # Skip invalid coordinates or zero/negative dimensions
            if w <= 0 or h <= 0:
                continue
                
            # Filter out tiny faces
            if w < min_face_size or h < min_face_size:
                continue
                
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = (xmin, ymin, w, h)
                
        return largest_face

    def close(self) -> None:
        """
        Closes the MediaPipe face detector instance.
        """
        self.detector.close()


def get_square_crop_coords(
    xmin: int, 
    ymin: int, 
    w: int, 
    h: int, 
    img_width: int, 
    img_height: int, 
    padding: float
) -> Tuple[int, int, int, int]:
    """
    Computes absolute coordinates for a square crop surrounding the face,
    incorporating padding and adjusting boundary coordinates if they overflow.
    
    Args:
        xmin (int): Left pixel coordinate.
        ymin (int): Top pixel coordinate.
        w (int): Face width.
        h (int): Face height.
        img_width (int): Source image width.
        img_height (int): Source image height.
        padding (float): Bounding box padding multiplier.
        
    Returns:
        Tuple[int, int, int, int]: (x1, y1, x2, y2) crop coordinates.
    """
    cx = xmin + w / 2.0
    cy = ymin + h / 2.0
    
    # Calculate padded box size
    padded_w = w * (1.0 + 2.0 * padding)
    padded_h = h * (1.0 + 2.0 * padding)
    
    # Use maximum of width and height to guarantee square crop (no distortion)
    side = max(padded_w, padded_h)
    
    # Define corners
    x1 = int(cx - side / 2.0)
    y1 = int(cy - side / 2.0)
    x2 = int(x1 + side)
    y2 = int(y1 + side)
    
    # Adjust for boundaries by shifting the crop area rather than just clipping
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_width:
        diff = x2 - img_width
        x1 = max(0, x1 - diff)
        x2 = img_width
    if y2 > img_height:
        diff = y2 - img_height
        y1 = max(0, y1 - diff)
        y2 = img_height
        
    return x1, y1, x2, y2


def process_image_list(
    image_paths: List[Path],
    output_dir: Path,
    config: AppConfig
) -> Tuple[int, int]:
    """
    Batch processes a list of frame images by detecting, cropping, and saving faces.
    
    Args:
        image_paths (List[Path]): List of frame paths.
        output_dir (Path): Output folder for cropped faces.
        config (AppConfig): Application configuration.
        
    Returns:
        Tuple[int, int]: (success_count, ignored_or_failed_count).
    """
    success_count = 0
    fail_count = 0
    
    detector = FaceDetector(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                logging.warning(f"Could not read image: {path}")
                fail_count += 1
                continue
                
            img_h, img_w, _ = img.shape
            face_box = detector.detect_largest_face(img, config.pipeline.min_face_size)
            
            if face_box is None:
                # Log frame-level exclusion (ignored if no face or tiny face)
                logging.debug(f"No valid face detected in {path.name}. Skipping frame.")
                fail_count += 1
                continue
                
            xmin, ymin, w, h = face_box
            x1, y1, x2, y2 = get_square_crop_coords(
                xmin, ymin, w, h, img_w, img_h, config.pipeline.face_padding
            )
            
            # Perform crop
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                logging.warning(f"Empty crop for image {path.name}")
                fail_count += 1
                continue
                
            # Resize
            resized = cv2.resize(
                crop, 
                config.pipeline.image_size, 
                interpolation=cv2.INTER_CUBIC
            )
            
            # Save cropped face
            output_path = output_dir / path.name
            success = cv2.imwrite(
                str(output_path), 
                resized, 
                [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            
            if success:
                success_count += 1
            else:
                logging.error(f"Failed to write cropped image to {output_path}")
                fail_count += 1
                
    finally:
        detector.close()
        
    return success_count, fail_count


def process_video_directory(args: Tuple[Path, Path, Path, AppConfig]) -> Dict[str, Any]:
    """
    Orchestrates face detection and cropping for all frames inside a video folder.
    
    Args:
        args (Tuple[Path, Path, Path, AppConfig]): Unpacked argument:
            (video_frames_dir, frames_root_dir, output_crops_root_dir, config)
            
    Returns:
        Dict[str, Any]: Processing metrics.
    """
    video_frames_dir, frames_root_dir, output_crops_root_dir, config = args
    start_time = time.time()
    
    # Identify video subdirectory name relative to frames root (maintaining real/fake splits)
    try:
        relative_dir = video_frames_dir.relative_to(frames_root_dir)
        output_dir = output_crops_root_dir / relative_dir
    except ValueError:
        output_dir = output_crops_root_dir / video_frames_dir.name
        
    # Discover frame files sorted
    frame_paths = sorted([
        p for p in video_frames_dir.glob("*.jpg")
        if p.name.startswith("frame_")
    ])
    
    success_count = 0
    fail_count = 0
    status = "failed"
    error_msg = ""
    
    # Read video frame metadata
    orig_meta: Dict[str, Any] = {}
    meta_file = video_frames_dir / "metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                orig_meta = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load metadata from {meta_file}: {e}")

    try:
        if not frame_paths:
            raise ValueError(f"No frame files found in {video_frames_dir}")
            
        success_count, fail_count = process_image_list(frame_paths, output_dir, config)
        status = "success"
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Failed to process faces in {video_frames_dir}: {error_msg}")
        # Clean up output directory on complete failure
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            
    elapsed = time.time() - start_time
    
    result_meta = {
        "video_name": orig_meta.get("video_name", video_frames_dir.name),
        "original_path": orig_meta.get("original_path", ""),
        "frames_dir": str(video_frames_dir.resolve()),
        "output_crops_dir": str(output_dir.resolve()),
        "total_input_frames": len(frame_paths),
        "successful_crops": success_count,
        "ignored_or_failed_crops": fail_count,
        "status": status,
        "error_message": error_msg,
        "duration_seconds": elapsed,
        "processed_at": datetime.now().isoformat()
    }
    
    if status == "success":
        try:
            with open(output_dir / "face_metadata.json", "w", encoding="utf-8") as f:
                json.dump(result_meta, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to write face_metadata.json in {output_dir}: {e}")
            
    return result_meta


def is_already_cropped(video_frames_dir: Path, frames_root_dir: Path, output_crops_root_dir: Path) -> bool:
    """
    Checks if face detection has already completed successfully for a video folder.
    """
    try:
        relative_dir = video_frames_dir.relative_to(frames_root_dir)
        output_dir = output_crops_root_dir / relative_dir
    except ValueError:
        output_dir = output_crops_root_dir / video_frames_dir.name
        
    face_meta_file = output_dir / "face_metadata.json"
    if not face_meta_file.exists():
        return False
        
    try:
        with open(face_meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("status") == "success"
    except Exception:
        return False


def run_face_detection(config: AppConfig) -> Dict[str, Any]:
    """
    Orchestrates recursive face detection, cropping, and resizing across all video folders.
    
    Args:
        config (AppConfig): Application configuration.
        
    Returns:
        Dict[str, Any]: Global face detection run statistics.
    """
    start_time = time.time()
    frames_root_dir = config.paths.processed_dir / "extracted_frames"
    output_crops_root_dir = config.paths.processed_dir / "face_crops"
    
    logging.info("Starting face detection processing stage...")
    
    # Download the MediaPipe Face Detection model in the main process to prevent race conditions in multiprocessing
    model_dir = config.paths.project_root / "ai" / "video" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "blaze_face_short_range.tflite"
    if not model_path.exists():
        logging.info(f"MediaPipe face detection model not found. Downloading in main process to {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        try:
            # Atomic download: download to temp path first, then rename
            temp_path = model_path.with_suffix(".tmp")
            urllib.request.urlretrieve(url, str(temp_path))
            temp_path.rename(model_path)
            logging.info("Model download completed successfully in main process.")
        except Exception as e:
            logging.critical(f"Failed to download MediaPipe face detection model in main process: {e}")
            raise e
            
    if not frames_root_dir.exists():
        logging.warning(f"Frames directory does not exist: {frames_root_dir}. Please run frame extraction first.")
        return {
            "total_directories": 0,
            "processed": 0,
            "skipped": 0,
            "success": 0,
            "failed": 0,
            "total_face_crops": 0,
            "elapsed_seconds": 0.0
        }
        
    # Find all subdirectories that contain frames (depth 2 for real/fake subdirs)
    video_dirs: List[Path] = []
    # Search for directories that have frame files inside them
    for path in frames_root_dir.rglob("*"):
        if path.is_dir() and any(path.glob("frame_*.jpg")):
            video_dirs.append(path)
            
    video_dirs = sorted(video_dirs)
    logging.info(f"Discovered {len(video_dirs)} video frame directories.")
    
    if not video_dirs:
        logging.warning("No frame directories were found for face detection.")
        return {
            "total_directories": 0,
            "processed": 0,
            "skipped": 0,
            "success": 0,
            "failed": 0,
            "total_face_crops": 0,
            "elapsed_seconds": time.time() - start_time
        }
        
    # Filter for resuming
    dirs_to_process: List[Path] = []
    skipped_count = 0
    for v_dir in video_dirs:
        if is_already_cropped(v_dir, frames_root_dir, output_crops_root_dir):
            skipped_count += 1
        else:
            dirs_to_process.append(v_dir)
            
    logging.info(f"Skipped {skipped_count} already processed directories. {len(dirs_to_process)} left to process.")
    
    if not dirs_to_process:
        logging.info("All video frame directories have already been processed for face detection.")
        return {
            "total_directories": len(video_dirs),
            "processed": 0,
            "skipped": skipped_count,
            "success": 0,
            "failed": 0,
            "total_face_crops": 0,
            "elapsed_seconds": time.time() - start_time
        }
        
    # Multiprocessing pool setups
    pool_args = [
        (v_dir, frames_root_dir, output_crops_root_dir, config)
        for v_dir in dirs_to_process
    ]
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Starting face detection pool with {num_workers} processes...")
    
    results: List[Dict[str, Any]] = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_video_directory, pool_args),
            total=len(pool_args),
            desc="Cropping faces from frames"
        ):
            results.append(result)
            
    # Compute run stats
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    total_crops = sum(r["successful_crops"] for r in results)
    
    elapsed_time = time.time() - start_time
    
    summary_stats = {
        "total_directories": len(video_dirs),
        "processed": len(dirs_to_process),
        "skipped": skipped_count,
        "success": success_count,
        "failed": failed_count,
        "total_face_crops": total_crops,
        "elapsed_seconds": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save statistics
    stats_file = config.paths.processed_dir / "face_detection_statistics.json"
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=4)
        logging.info(f"Face detection statistics saved to {stats_file}")
    except Exception as e:
        logging.error(f"Failed to write face detection statistics: {e}")
        
    logging.info(
        f"Face detection complete. Success: {success_count}, Failed: {failed_count}. "
        f"Cropped {total_crops} faces in {elapsed_time:.2f} seconds."
    )
    
    return summary_stats


if __name__ == "__main__":
    # Test script execution
    app_config = get_default_config()
    setup_logging(app_config)
    logging.info("Starting dry-run verification of face_detector.py")
    
    run_face_detection(app_config)
