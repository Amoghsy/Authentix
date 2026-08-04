"""
run_preprocessing.py

This is the main orchestrator script for the Authentix video preprocessing pipeline.
It runs the full sequence of steps:
1. Frame Extraction
2. Face Detection, Padding, Cropping, and Scaling
3. Dataset Stratified Splitting and Building

It measures execution times, consolidates individual stage reports, and generates a
detailed final summary report of the dataset preparation process.
"""

import argparse
from datetime import datetime
import json
import logging
import sys
from pathlib import Path
import time
from typing import Dict, Any, Optional

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import (
    AppConfig, 
    get_default_config, 
    setup_directories, 
    setup_logging, 
    validate_config,
    PipelineConfig,
    SplitConfig
)
from ai.video.preprocessing.frame_extractor import run_frame_extraction
from ai.video.preprocessing.face_detector import run_face_detection
from ai.video.preprocessing.dataset_builder import build_split_dataset


def generate_final_report(
    config: AppConfig,
    extraction_stats: Dict[str, Any],
    detection_stats: Dict[str, Any],
    builder_stats: Optional[Dict[str, Any]],
    total_duration: float
) -> Path:
    """
    Consolidates preprocessing pipeline outputs and writes a detailed Markdown report.
    
    Args:
        config (AppConfig): Active application configuration.
        extraction_stats (Dict[str, Any]): Metrics from frame extraction.
        detection_stats (Dict[str, Any]): Metrics from face detection.
        builder_stats (Optional[Dict[str, Any]]): Metrics from dataset splits.
        total_duration (float): Total pipeline execution time in seconds.
        
    Returns:
        Path: Path to the generated report file.
    """
    report_file = config.paths.dataset_dir / "preprocessing_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format split ratio string
    splits_str = f"Train={config.split.train * 100:.0f}%, Val={config.split.val * 100:.0f}%, Test={config.split.test * 100:.0f}%"
    
    report_content = f"""# Authentix Preprocessing Pipeline Final Report

**Generated At:** {timestamp}  
**Total Pipeline Execution Time:** {total_duration:.2f} seconds ({total_duration / 60.0:.2f} minutes)  
**Status:** {"SUCCESS" if builder_stats and "error" not in builder_stats else "PARTIAL / FAILED"}

---

## 1. Pipeline Settings & Configurations
- **Project Root Directory:** `{config.paths.project_root}`
- **Dataset Input (Raw) Directory:** `{config.paths.raw_dir}`
- **Dataset Output (Processed) Directory:** `{config.paths.dataset_dir}`
- **Random Seed:** `{config.random_seed}`
- **Target Frame Rate (FPS):** `{config.pipeline.fps} Hz`
- **Output Bounding Box Size:** `{config.pipeline.image_size[0]}x{config.pipeline.image_size[1]}`
- **Face Crop Bounding Box Padding:** `{config.pipeline.face_padding * 100:.1f}%`
- **MediaPipe Minimum Confidence:** `{config.pipeline.mp_detection_confidence}`
- **Configured Splits:** `{splits_str}`

---

## 2. Phase 1: Frame Extraction Statistics
- **Discovered Videos in Raw Directory:** {extraction_stats.get("total_discovered", 0)}
- **Videos Processed in This Run:** {extraction_stats.get("processed", 0)}
- **Videos Resumed/Skipped:** {extraction_stats.get("skipped", 0)}
- **Extractions Succeeded:** {extraction_stats.get("success", 0)}
- **Extractions Failed (Corrupted/Errors):** {extraction_stats.get("failed", 0)}
- **Total Raw Frames Extracted:** {extraction_stats.get("total_frames_extracted", 0)}
- **Execution Duration:** {extraction_stats.get("elapsed_seconds", 0.0):.2f} seconds

---

## 3. Phase 2: Face Detection & Cropping Statistics
- **Directories Scanned:** {detection_stats.get("total_directories", 0)}
- **Directories Processed in This Run:** {detection_stats.get("processed", 0)}
- **Directories Resumed/Skipped:** {detection_stats.get("skipped", 0)}
- **Detection Run Succeeded:** {detection_stats.get("success", 0)}
- **Detection Run Failed:** {detection_stats.get("failed", 0)}
- **Total Cropped Face Images Generated:** {detection_stats.get("total_face_crops", 0)}
- **Execution Duration:** {detection_stats.get("elapsed_seconds", 0.0):.2f} seconds

---

## 4. Phase 3: Final Dataset Split Summary
"""

    if builder_stats and "error" not in builder_stats:
        video_counts = builder_stats.get("video_counts", {})
        frame_counts = builder_stats.get("frame_counts", {})
        
        report_content += f"""
### Video Distribution
| Split | Real Videos | Fake Videos | Total Videos |
| :--- | :---: | :---: | :---: |
| **Train** | {video_counts.get("train", {}).get("real", 0)} | {video_counts.get("train", {}).get("fake", 0)} | {video_counts.get("train", {}).get("total", 0)} |
| **Validation** | {video_counts.get("validation", {}).get("real", 0)} | {video_counts.get("validation", {}).get("fake", 0)} | {video_counts.get("validation", {}).get("total", 0)} |
| **Test** | {video_counts.get("test", {}).get("real", 0)} | {video_counts.get("test", {}).get("fake", 0)} | {video_counts.get("test", {}).get("total", 0)} |
| **Overall Total** | **{video_counts.get("overall", {}).get("real", 0)}** | **{video_counts.get("overall", {}).get("fake", 0)}** | **{video_counts.get("overall", {}).get("total", 0)}** |

### Frame Distribution (Cropped Face Images)
| Split | Real Frames | Fake Frames | Total Frames |
| :--- | :---: | :---: | :---: |
| **Train** | {frame_counts.get("train", {}).get("real", 0)} | {frame_counts.get("train", {}).get("fake", 0)} | {frame_counts.get("train", {}).get("total", 0)} |
| **Validation** | {frame_counts.get("validation", {}).get("real", 0)} | {frame_counts.get("validation", {}).get("fake", 0)} | {frame_counts.get("validation", {}).get("total", 0)} |
| **Test** | {frame_counts.get("test", {}).get("real", 0)} | {frame_counts.get("test", {}).get("fake", 0)} | {frame_counts.get("test", {}).get("total", 0)} |
| **Overall Total** | **{frame_counts.get("overall", {}).get("real", 0)}** | **{frame_counts.get("overall", {}).get("fake", 0)}** | **{frame_counts.get("overall", {}).get("total", 0)}** |

---
*The dataset is now structured and ready for PyTorch MobileNetV3 Training.*
"""
    else:
        err = builder_stats.get("error", "No data split occurred due to previous stage warning/failure.") if builder_stats else "N/A"
        report_content += f"""
> [!WARNING]
> Dataset Split was skipped or failed.
> Reason: `{err}`
"""

    try:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        logging.info(f"Saved pipeline execution report to: {report_file}")
    except Exception as e:
        logging.error(f"Failed to write final pipeline report to {report_file}: {e}")
        
    return report_file


def count_successful_videos(face_crops_dir: Path, class_filter: Optional[str] = None) -> int:
    """
    Counts the number of video directories inside face_crops that contain
    successful crops metadata, optionally filtered by class.
    """
    if not face_crops_dir.exists():
        return 0
    count = 0
    for path in face_crops_dir.rglob("face_metadata.json"):
        try:
            video_dir = path.parent
            relative_path = video_dir.relative_to(face_crops_dir)
            first_part = relative_path.parts[0].lower()
            
            if first_part in {"real", "original", "youtube", "original_sequences"}:
                label = "real"
            else:
                label = "fake"
                
            if class_filter is not None and label != class_filter.lower():
                continue
                
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if meta.get("status") == "success" and meta.get("successful_crops", 0) > 0:
                    count += 1
        except Exception:
            pass
    return count


def run_pipeline(config: AppConfig, target_success: Optional[int] = None) -> None:
    """
    Executes the entire video preprocessing pipeline.
    
    Args:
        config (AppConfig): The application configuration.
        target_success (Optional[int]): Target number of successful videos to produce.
    """
    logging.info("======================================================================")
    logging.info("AUTHENTIX MULTIMODAL VIDEO PREPROCESSING PIPELINE INITIALIZED")
    logging.info("======================================================================")
    
    start_time = time.time()
    
    # 0. Setup and Validate configuration
    validate_config(config)
    setup_directories(config)
    
    face_crops_dir = config.paths.processed_dir / "face_crops"
    
    extraction_stats = {}
    detection_stats = {}
    builder_stats = None
    
    if target_success is not None and target_success > 0:
        logging.info(f"Targeting a total of {target_success} successful videos with detected faces.")
        
        while True:
            current_success = count_successful_videos(face_crops_dir, config.pipeline.class_filter)
            logging.info(f"Current progress: {current_success} / {target_success} successful videos.")
            
            if current_success >= target_success:
                logging.info(f"Target of {target_success} successful videos met! Proceeding to Dataset split building.")
                break
                
            needed = target_success - current_success
            # Estimate how many raw videos to process (e.g. 1.2x for real class, 1.8x for fake class)
            multiplier = 1.2 if (config.pipeline.class_filter and config.pipeline.class_filter.lower() == "real") else 1.8
            batch_size = max(50, min(400, int(needed * multiplier)))
            logging.info(f"Need {needed} more successful videos. Processing next batch of {batch_size} videos...")
            
            # AppConfig is frozen, so we recreate it with overridden max_videos
            from dataclasses import replace
            new_pipeline = replace(config.pipeline, max_videos=batch_size)
            iter_config = replace(config, pipeline=new_pipeline)
            
            # Run Stage 1 (Frame Extraction) on this batch
            logging.info("--- STAGE 1 (Batch): Frame Extraction ---")
            stage_start = time.time()
            extraction_stats = run_frame_extraction(iter_config)
            extraction_duration = time.time() - stage_start
            logging.info(f"Batch Stage 1 completed in {extraction_duration:.2f} seconds.")
            
            if extraction_stats.get("processed", 0) == 0:
                logging.warning("All raw videos have been processed. Cannot process more videos.")
                break
                
            # Run Stage 2 (Face Detection) on this batch
            logging.info("--- STAGE 2 (Batch): Face Detection & Bounding Box Cropping ---")
            stage_start = time.time()
            detection_stats = run_face_detection(iter_config)
            detection_duration = time.time() - stage_start
            logging.info(f"Batch Stage 2 completed in {detection_duration:.2f} seconds.")
            
        # Run Stage 3 (Dataset Builder) on all accumulated successful face crops
        logging.info("--- STAGE 3: Dataset Builder & Split Partitioning ---")
        stage_start = time.time()
        # Ensure we have directories to build splits
        current_success = count_successful_videos(face_crops_dir)
        if current_success > 0:
            builder_stats = build_split_dataset(config)
        else:
            logging.warning("No face crops available. Skipping Dataset Split step.")
        builder_duration = time.time() - stage_start
        logging.info(f"Stage 3 completed in {builder_duration:.2f} seconds.")
        
    else:
        # 1. Frame Extraction
        logging.info("--- STAGE 1: Frame Extraction ---")
        stage_start = time.time()
        extraction_stats = run_frame_extraction(config)
        extraction_duration = time.time() - stage_start
        logging.info(f"Stage 1 completed in {extraction_duration:.2f} seconds.")
        
        # 2. Face Detection & Cropping
        logging.info("--- STAGE 2: Face Detection & Bounding Box Cropping ---")
        stage_start = time.time()
        detection_stats = run_face_detection(config)
        detection_duration = time.time() - stage_start
        logging.info(f"Stage 2 completed in {detection_duration:.2f} seconds.")
        
        # 3. Dataset Building (Stratified splits by video)
        logging.info("--- STAGE 3: Dataset Builder & Split Partitioning ---")
        stage_start = time.time()
        builder_stats = None
        face_dirs = []
        if face_crops_dir.exists():
            face_dirs = [d for d in face_crops_dir.rglob("*") if d.is_dir() and any(d.glob("*.jpg"))]
        if face_dirs:
            builder_stats = build_split_dataset(config)
        else:
            logging.warning("No face crops available. Skipping Dataset Split step.")
        builder_duration = time.time() - stage_start
        logging.info(f"Stage 3 completed in {builder_duration:.2f} seconds.")
        
    total_duration = time.time() - start_time
    logging.info("======================================================================")
    logging.info(f"PIPELINE COMPLETED. Total elapsed time: {total_duration:.2f} seconds.")
    logging.info("======================================================================")
    
    # 4. Generate final reports
    report_path = generate_final_report(
        config,
        extraction_stats,
        detection_stats,
        builder_stats,
        total_duration
    )
    
    # Print the report content in terminal for user visualization
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            print("\n" + f.read())
    except Exception as e:
        logging.error(f"Could not read final report for stdout printing: {e}")


def main() -> None:
    """
    Parses command line arguments to override defaults and launches preprocessing.
    """
    default_config = get_default_config()
    
    parser = argparse.ArgumentParser(
        description="Authentix - Multimodality Deepfake Detection Preprocessing Pipeline"
    )
    parser.add_argument(
        "--raw-dir", 
        type=str, 
        help=f"Override path for input raw videos. Default: {default_config.paths.raw_dir}"
    )
    parser.add_argument(
        "--fps", 
        type=float, 
        help=f"Override FPS sampling rate. Default: {default_config.pipeline.fps}"
    )
    parser.add_argument(
        "--image-size", 
        type=int, 
        nargs=2, 
        metavar=("HEIGHT", "WIDTH"),
        help=f"Override target crop output size. Default: {default_config.pipeline.image_size}"
    )
    parser.add_argument(
        "--face-padding", 
        type=float, 
        help=f"Override face cropping box padding multiplier. Default: {default_config.pipeline.face_padding}"
    )
    parser.add_argument(
        "--splits", 
        type=float, 
        nargs=3, 
        metavar=("TRAIN", "VAL", "TEST"),
        help=f"Override train, val, test ratios (must sum to 1.0). Default: {default_config.split.train}, {default_config.split.val}, {default_config.split.test}"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help=f"Override seed for split shuffling. Default: {default_config.random_seed}"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help=f"Limit the total number of videos to process. Default: {default_config.pipeline.max_videos}"
    )
    parser.add_argument(
        "--target-success",
        type=int,
        help="Target number of successful videos (with faces) to produce."
    )
    parser.add_argument(
        "--class-filter",
        type=str,
        choices=["real", "fake"],
        help="Filter videos to process only 'real' or 'fake' classes."
    )
    
    args = parser.parse_args()
    
    # Override configuration attributes dynamically based on user CLI args
    raw_dir_path = Path(args.raw_dir) if args.raw_dir else default_config.paths.raw_dir
    fps = args.fps if args.fps is not None else default_config.pipeline.fps
    img_size = tuple(args.image_size) if args.image_size else default_config.pipeline.image_size
    face_padding = args.face_padding if args.face_padding is not None else default_config.pipeline.face_padding
    seed = args.seed if args.seed is not None else default_config.random_seed
    max_videos = args.max_videos if args.max_videos is not None else default_config.pipeline.max_videos
    target_success = args.target_success
    class_filter = args.class_filter if args.class_filter is not None else default_config.pipeline.class_filter
    
    train_split, val_split, test_split = default_config.split.train, default_config.split.val, default_config.split.test
    if args.splits:
        train_split, val_split, test_split = args.splits
        
    # Reassemble config objects
    path_config = PathConfig = default_config.paths
    if args.raw_dir:
        path_config = PathConfig = default_config.paths.__class__(
            project_root=default_config.paths.project_root,
            video_root=default_config.paths.video_root,
            dataset_dir=default_config.paths.dataset_dir,
            raw_dir=raw_dir_path,
            processed_dir=default_config.paths.processed_dir,
            train_dir=default_config.paths.train_dir,
            val_dir=default_config.paths.val_dir,
            test_dir=default_config.paths.test_dir,
            logs_dir=default_config.paths.logs_dir,
            checkpoints_dir=default_config.paths.checkpoints_dir
        )
        
    pipeline_config = PipelineConfig(
        image_size=img_size,
        fps=fps,
        supported_extensions=default_config.pipeline.supported_extensions,
        min_face_size=default_config.pipeline.min_face_size,
        face_padding=face_padding,
        mp_detection_confidence=default_config.pipeline.mp_detection_confidence,
        max_videos=max_videos,
        class_filter=class_filter
    )
    
    split_config = SplitConfig(
        train=train_split,
        val=val_split,
        test=test_split
    )
    
    config = AppConfig(
        paths=path_config,
        pipeline=pipeline_config,
        split=split_config,
        random_seed=seed
    )
    
    # Configure logging and run
    setup_logging(config)
    run_pipeline(config, target_success)


if __name__ == "__main__":
    main()
