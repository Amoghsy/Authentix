"""
frame_extractor.py

This module contains the FrameExtractor class and supporting functions for recursively
discovering videos in raw folders, extracting video frames at a configurable FPS,
preserving metadata, and handling resume/error logic. Multiprocessing is utilized
for high performance.
"""

from datetime import datetime
import json
import logging
import multiprocessing
import sys
from pathlib import Path
import shutil
import time
from typing import Dict, Any, List, Set, Tuple, Optional

import cv2
from tqdm import tqdm

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging


def discover_videos(input_dir: Path, extensions: Set[str]) -> List[Path]:
    """
    Recursively scans the directory for videos matching the supported extensions.
    
    Args:
        input_dir (Path): Root directory to search.
        extensions (Set[str]): Set of lowercase extensions, e.g. {'.mp4', '.avi'}.
        
    Returns:
        List[Path]: A list of paths to found video files.
    """
    video_paths: List[Path] = []
    if not input_dir.exists():
        logging.warning(f"Input directory does not exist: {input_dir}")
        return video_paths

    for file_path in input_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            video_paths.append(file_path)
            
    return sorted(video_paths)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    target_fps: float
) -> Dict[str, Any]:
    """
    Extracts frames from a single video at a target FPS, saves them as JPG,
    and returns metadata/statistics dictionary. Cleans up on failure.
    
    Args:
        video_path (Path): Path to the source video.
        output_dir (Path): Target directory to save extracted frames.
        target_fps (float): Number of frames to extract per second.
        
    Returns:
        Dict[str, Any]: Metadata detailing the extraction results.
    """
    start_time = time.time()
    extracted_count = 0
    status = "failed"
    error_msg = ""
    orig_fps = 0.0
    width = 0
    height = 0
    total_frames = 0
    duration_sec = 0.0

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise IOError("Could not open video file. File might be corrupted or has unsupported codec.")
            
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if orig_fps <= 0 or total_frames <= 0:
            raise IOError(f"Invalid video metadata: FPS={orig_fps}, total_frames={total_frames}")
            
        duration_sec = total_frames / orig_fps
        
        # We extract frames based on elapsed time to avoid roundoff errors.
        time_step = 1.0 / target_fps
        next_extraction_time = 0.0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            current_time = frame_idx / orig_fps
            if current_time >= next_extraction_time:
                # Save the frame
                frame_name = f"frame_{extracted_count:05d}.jpg"
                frame_path = output_dir / frame_name
                
                # Write with high JPEG quality for training
                success = cv2.imwrite(
                    str(frame_path), 
                    frame, 
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                )
                if not success:
                    raise IOError(f"Failed to write frame at {frame_path}")
                    
                extracted_count += 1
                next_extraction_time += time_step
                
            frame_idx += 1

        status = "success"
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error extracting frames from {video_path}: {error_msg}")
        # Clean up output directory on failure so it is not marked as successful
        if output_dir.exists():
            shutil.rmtree(output_dir)
            
    finally:
        cap.release()

    elapsed = time.time() - start_time
    
    metadata = {
        "video_name": video_path.name,
        "original_path": str(video_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "original_fps": orig_fps,
        "target_fps": target_fps,
        "width": width,
        "height": height,
        "total_frames_in_video": total_frames,
        "duration_seconds": duration_sec,
        "extracted_frames_count": extracted_count,
        "status": status,
        "error_message": error_msg,
        "extraction_duration_seconds": elapsed,
        "processed_at": datetime.now().isoformat()
    }
    
    # Save video-specific metadata file inside output_dir if successful
    if status == "success":
        try:
            with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logging.error(f"Failed to write metadata.json to {output_dir}: {e}")
            
    return metadata


def process_video_wrapper(args: Tuple[Path, Path, Path, float]) -> Dict[str, Any]:
    """
    Unpacks arguments to invoke frame extraction. Useful for multiprocessing pool mapping.
    
    Args:
        args (Tuple[Path, Path, Path, float]): Packed args containing:
            (video_path, raw_dir, output_root_dir, target_fps)
            
    Returns:
        Dict[str, Any]: Resulting metadata dictionary.
    """
    video_path, raw_dir, output_root_dir, target_fps = args
    
    # Calculate the target output folder based on relative structure
    try:
        relative_path = video_path.relative_to(raw_dir)
        # Target directory keeps same structure but replaces file name with a folder of frames
        output_dir = output_root_dir / relative_path.parent / video_path.stem
    except ValueError:
        # Fallback if video is not inside raw_dir
        output_dir = output_root_dir / video_path.stem
        
    return extract_frames_from_video(video_path, output_dir, target_fps)


def is_already_processed(video_path: Path, raw_dir: Path, output_root_dir: Path) -> bool:
    """
    Checks if a video was already processed successfully by searching for a valid metadata.json.
    
    Args:
        video_path (Path): Path to the video file.
        raw_dir (Path): Root raw dataset path.
        output_root_dir (Path): Output directory for frame extraction.
        
    Returns:
        bool: True if already processed successfully, False otherwise.
    """
    try:
        relative_path = video_path.relative_to(raw_dir)
        output_dir = output_root_dir / relative_path.parent / video_path.stem
    except ValueError:
        output_dir = output_root_dir / video_path.stem
        
    metadata_file = output_dir / "metadata.json"
    if not metadata_file.exists():
        return False
        
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("status") == "success"
    except Exception:
        # If metadata.json is corrupted, re-extract
        return False


def get_video_class(video_path: Path, raw_dir: Path) -> str:
    """
    Identifies the class (real or fake) of a video path based on its first parent
    relative to the raw videos directory.
    """
    try:
        rel = video_path.relative_to(raw_dir)
        first_part = rel.parts[0].lower()
    except Exception:
        first_part = ""
    if first_part in {"real", "original", "youtube", "original_sequences"}:
        return "real"
    return "fake"


def run_frame_extraction(config: AppConfig) -> Dict[str, Any]:
    """
    Orchestrates the discovery, filtering, and parallel extraction of frames.
    
    Args:
        config (AppConfig): Application configuration parameters.
        
    Returns:
        Dict[str, Any]: Run statistics.
    """
    start_time = time.time()
    raw_dir = config.paths.raw_dir
    output_root_dir = config.paths.processed_dir / "extracted_frames"
    
    logging.info("Starting recursive video discovery...")
    all_videos = discover_videos(raw_dir, config.pipeline.supported_extensions)
    logging.info(f"Discovered {len(all_videos)} videos in total.")

    if not all_videos:
        logging.warning("No videos were found for processing.")
        return {
            "total_discovered": 0,
            "processed": 0,
            "skipped": 0,
            "success": 0,
            "failed": 0,
            "total_frames_extracted": 0,
            "elapsed_seconds": time.time() - start_time
        }

    # Filter out already processed videos for resumption capability
    unprocessed_videos: List[Path] = []
    skipped_count = 0
    for video in all_videos:
        if is_already_processed(video, raw_dir, output_root_dir):
            skipped_count += 1
        else:
            unprocessed_videos.append(video)
            
    logging.info(f"Skipped {skipped_count} already processed videos. {len(unprocessed_videos)} unprocessed left.")
    
    # Filter by class_filter if configured
    if config.pipeline.class_filter is not None:
        c_filter = config.pipeline.class_filter.lower()
        logging.info(f"Filtering unprocessed videos to only process class: {c_filter}")
        unprocessed_videos = [
            v for v in unprocessed_videos
            if get_video_class(v, raw_dir) == c_filter
        ]
        logging.info(f"After class filtering: {len(unprocessed_videos)} unprocessed videos left to process.")
    
    # Limit number of videos to process if config.pipeline.max_videos is set
    videos_to_process = list(unprocessed_videos)
    if config.pipeline.max_videos is not None and config.pipeline.max_videos > 0:
        max_v = config.pipeline.max_videos
        logging.info(f"Limiting this batch run to process a maximum of {max_v} new videos (balanced by subdirectory)...")
        
        # Group videos by their subfolder relative to raw_dir to ensure class balance
        grouped_videos: Dict[str, List[Path]] = {}
        for video in unprocessed_videos:
            try:
                rel_part = video.relative_to(raw_dir).parts[0]
            except Exception:
                rel_part = "unknown"
            grouped_videos.setdefault(rel_part, []).append(video)
            
        num_groups = len(grouped_videos)
        if num_groups > 0:
            limit_per_group = max_v // num_groups
            remainder = max_v % num_groups
            
            selected_videos: List[Path] = []
            import random
            for idx, (group_name, group_list) in enumerate(sorted(grouped_videos.items())):
                # Shuffle using the seeded RNG for reproducibility
                rng = random.Random(config.random_seed)
                shuffled_group = list(group_list)
                rng.shuffle(shuffled_group)
                
                limit = limit_per_group + (1 if idx < remainder else 0)
                selected_videos.extend(shuffled_group[:limit])
                
            videos_to_process = sorted(selected_videos)
            logging.info(f"Batch run limited to {len(videos_to_process)} videos (sampled from groups: {list(grouped_videos.keys())}).")

    if not videos_to_process:
        logging.info("All videos have already been processed.")
        # Load accumulated counts if possible, otherwise return basic statistics
        return {
            "total_discovered": len(all_videos),
            "processed": 0,
            "skipped": skipped_count,
            "success": 0,
            "failed": 0,
            "total_frames_extracted": 0,
            "elapsed_seconds": time.time() - start_time
        }

    # Prepare pool arguments
    pool_args = [
        (video, raw_dir, output_root_dir, config.pipeline.fps)
        for video in videos_to_process
    ]
    
    # Run multiprocessing extraction
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Starting multiprocessing pool with {num_workers} workers...")
    
    results: List[Dict[str, Any]] = []
    
    # We display a progress bar for video-level extraction completion
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_video_wrapper, pool_args),
            total=len(pool_args),
            desc="Extracting video frames"
        ):
            results.append(result)

    # Compute execution statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    total_frames = sum(r["extracted_frames_count"] for r in results)
    
    elapsed_time = time.time() - start_time
    
    summary_stats = {
        "total_discovered": len(all_videos),
        "processed": len(videos_to_process),
        "skipped": skipped_count,
        "success": success_count,
        "failed": failed_count,
        "total_frames_extracted": total_frames,
        "elapsed_seconds": elapsed_time,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save statistics file
    stats_file = config.paths.processed_dir / "extraction_statistics.json"
    try:
        config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=4)
        logging.info(f"Extraction run statistics saved to {stats_file}")
    except Exception as e:
        logging.error(f"Failed to write global extraction stats to {stats_file}: {e}")

    logging.info(
        f"Extraction complete. Success: {success_count}, Failed: {failed_count}. "
        f"Extracted {total_frames} frames in {elapsed_time:.2f} seconds."
    )
    
    return summary_stats


if __name__ == "__main__":
    # Test script execution flow
    app_config = get_default_config()
    setup_logging(app_config)
    
    logging.info("Starting dry-run verification of frame_extractor.py")
    
    # Create a small dummy video in the raw folder to test if raw folder is empty
    raw_dir = app_config.paths.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    run_frame_extraction(app_config)
