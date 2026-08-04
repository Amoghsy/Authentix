"""
dataset_builder.py

This module orchestrates splitting the cropped faces dataset into train, validation,
and test sets. The split is performed strictly at the video level (never splitting frames
of the same video across splits) to prevent data leakage. It generates flat split folders
with renamed files to prevent naming collisions, outputs dataset statistics, a CSV manifest,
a label mapping, and a summary report.
"""

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import json
import logging
import random
import shutil
import sys
from pathlib import Path
import time
from typing import Dict, Any, List, Tuple

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging


def determine_class_label(video_dir: Path, face_crops_root: Path) -> str:
    """
    Determines if a video folder belongs to the 'real' or 'fake' class
    based on relative folder structure or path naming heuristics.
    
    Args:
        video_dir (Path): The folder containing cropped face frames.
        face_crops_root (Path): Root folder containing all face crops.
        
    Returns:
        str: 'real', 'fake', or 'unknown'
    """
    try:
        relative_path = video_dir.relative_to(face_crops_root)
        first_part = relative_path.parts[0].lower()
    except (ValueError, IndexError):
        first_part = ""
        
    # Check direct relative path prefix
    if first_part in {"real", "original", "youtube", "original_sequences"}:
        return "real"
    if first_part in {
        "fake", "manipulated", "manipulated_sequences", "deepfakes", 
        "face2face", "faceswap", "faceshifter", "neuraltextures", 
        "deepfakedetection"
    }:
        return "fake"
        
    # Fallback heuristic: check substring matches in absolute path
    path_str = str(video_dir.resolve()).lower()
    if "original" in path_str or "real" in path_str or "youtube" in path_str:
        return "real"
    if any(k in path_str for k in [
        "manipulated", "fake", "deepfake", "face2face", "faceswap", 
        "faceshifter", "neuraltextures"
    ]):
        return "fake"
        
    return "unknown"


def copy_frame_file(args: Tuple[Path, Path, str, str, str]) -> Dict[str, Any]:
    """
    Copies a single frame to its target split/class path with a unique filename,
    returning a manifest dictionary row.
    
    Args:
        args (Tuple[Path, Path, str, str, str]): Unpacked parameters:
            (src_frame_path, dest_class_dir, video_name, label, split)
            
    Returns:
        Dict[str, Any]: Manifest record fields.
    """
    src_frame_path, dest_class_dir, video_name, label, split = args
    
    # Rename frame to prevent collisions: video_name_frame_idx.jpg
    new_filename = f"{video_name}_{src_frame_path.name}"
    dest_path = dest_class_dir / new_filename
    
    # Copy file preserving metadata
    shutil.copy2(src_frame_path, dest_path)
    
    # Calculate path relative to dataset_dir for the manifest
    # (dest_path is under project_root/ai/video/dataset/split/class/new_filename)
    # dataset_dir is project_root/ai/video/dataset
    try:
        relative_manifest_path = dest_path.relative_to(dest_class_dir.parents[1])
    except ValueError:
        relative_manifest_path = dest_path
        
    return {
        "file_path": str(relative_manifest_path.as_posix()),
        "video_name": video_name,
        "label": label,
        "split": split,
        "original_frame_name": src_frame_path.name
    }


def split_class_videos(
    video_dirs: List[Path], 
    split_config: Any, 
    seed: int
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Splits a list of video directories into train, validation, and test sets.
    
    Args:
        video_dirs (List[Path]): List of video directory paths.
        split_config: SplitConfig object from AppConfig.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Tuple[List[Path], List[Path], List[Path]]: (train, validation, test) splits.
    """
    rng = random.Random(seed)
    shuffled = list(video_dirs)
    rng.shuffle(shuffled)
    
    n = len(shuffled)
    if n == 0:
        return [], [], []
    if n == 1:
        return shuffled, [], []
    if n == 2:
        return [shuffled[0]], [shuffled[1]], []
        
    # Split calculations
    n_train = max(1, int(round(n * split_config.train)))
    n_val = max(1, int(round(n * split_config.val)))
    
    # Adjust in case rounding goes over bounds
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
        
    train_dirs = shuffled[:n_train]
    val_dirs = shuffled[n_train:n_train + n_val]
    test_dirs = shuffled[n_train + n_val:]
    
    return train_dirs, val_dirs, test_dirs


def build_split_dataset(config: AppConfig) -> Dict[str, Any]:
    """
    Orchestrates dataset grouping, split partition, copying, and manifest generation.
    
    Args:
        config (AppConfig): Application configuration.
        
    Returns:
        Dict[str, Any]: Execution statistics.
    """
    start_time = time.time()
    face_crops_root = config.paths.processed_dir / "face_crops"
    dataset_root = config.paths.dataset_dir
    
    logging.info("Starting dataset builder stage...")
    if not face_crops_root.exists():
        logging.warning(f"Face crops directory does not exist: {face_crops_root}. Run face detection first.")
        return {"status": "failed", "error": "No face crops found"}
        
    # 1. Discover all processed video folders containing face crops
    all_video_dirs: List[Path] = []
    for path in face_crops_root.rglob("*"):
        if path.is_dir() and any(path.glob("*.jpg")):
            all_video_dirs.append(path)
            
    all_video_dirs = sorted(all_video_dirs)
    logging.info(f"Discovered {len(all_video_dirs)} directories with face crops.")
    
    if not all_video_dirs:
        logging.warning("No face crops directories were found for dataset construction.")
        return {"status": "failed", "error": "No face crops directories found"}
        
    # 2. Categorize videos into classes
    real_videos: List[Path] = []
    fake_videos: List[Path] = []
    unknown_videos: List[Path] = []
    
    for v_dir in all_video_dirs:
        label = determine_class_label(v_dir, face_crops_root)
        if label == "real":
            real_videos.append(v_dir)
        elif label == "fake":
            fake_videos.append(v_dir)
        else:
            unknown_videos.append(v_dir)
            
    logging.info(f"Class counts - Real: {len(real_videos)}, Fake: {len(fake_videos)}, Unknown: {len(unknown_videos)}")
    if unknown_videos:
        logging.warning(f"Found {len(unknown_videos)} video directories with unknown labels: {unknown_videos}")
        # Add unknown videos to fake or log as skipped. Let's skip them to remain clean.
        
    # 3. Perform class-wise stratified split at video level
    real_train, real_val, real_test = split_class_videos(real_videos, config.split, config.random_seed)
    fake_train, fake_val, fake_test = split_class_videos(fake_videos, config.split, config.random_seed)
    
    # 4. Clear/Setup clean directories for final splits
    split_mappings = {
        "train": {"real": real_train, "fake": fake_train},
        "validation": {"real": real_val, "fake": fake_val},
        "test": {"real": real_test, "fake": fake_test}
    }
    
    split_dirs = {
        "train": config.paths.train_dir,
        "validation": config.paths.val_dir,
        "test": config.paths.test_dir
    }
    
    for split_name, class_dirs in split_mappings.items():
        for class_name in ["real", "fake"]:
            target_class_dir = split_dirs[split_name] / class_name
            if target_class_dir.exists():
                logging.info(f"Clearing existing directory: {target_class_dir}")
                shutil.rmtree(target_class_dir)
            target_class_dir.mkdir(parents=True, exist_ok=True)
            
    # 5. Populate splits in parallel
    copy_tasks: List[Tuple[Path, Path, str, str, str]] = []
    
    # Collect copy tasks
    for split_name, class_splits in split_mappings.items():
        for class_name, src_dirs in class_splits.items():
            dest_class_dir = split_dirs[split_name] / class_name
            for src_dir in src_dirs:
                video_name = src_dir.name
                frames = sorted(src_dir.glob("*.jpg"))
                for frame_path in frames:
                    copy_tasks.append((frame_path, dest_class_dir, video_name, class_name, split_name))
                    
    logging.info(f"Total frame files to copy: {len(copy_tasks)}")
    
    manifest_records: List[Dict[str, Any]] = []
    
    # Copy frames in parallel using ThreadPoolExecutor (I/O bound task)
    start_copy = time.time()
    num_threads = 16
    logging.info(f"Copying files using ThreadPoolExecutor with {num_threads} threads...")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(copy_frame_file, copy_tasks))
        manifest_records.extend(results)
        
    logging.info(f"File copy finished in {time.time() - start_copy:.2f} seconds.")
    
    # Sort manifest records by split, label, file path
    manifest_records.sort(key=lambda x: (x["split"], x["label"], x["file_path"]))
    
    # 6. Generate Manifest CSV
    manifest_csv_file = dataset_root / "dataset_manifest.csv"
    try:
        with open(manifest_csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file_path", "video_name", "label", "split", "original_frame_name"])
            for r in manifest_records:
                writer.writerow([r["file_path"], r["video_name"], r["label"], r["split"], r["original_frame_name"]])
        logging.info(f"Saved dataset manifest CSV to {manifest_csv_file}")
    except Exception as e:
        logging.error(f"Failed to save manifest CSV: {e}")
        
    # 7. Generate Label Mapping JSON
    label_mapping = {"real": 0, "fake": 1}
    label_mapping_file = dataset_root / "label_mapping.json"
    try:
        with open(label_mapping_file, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, indent=4)
        logging.info(f"Saved label mapping to {label_mapping_file}")
    except Exception as e:
        logging.error(f"Failed to save label mapping: {e}")
        
    # 8. Generate Statistics
    def count_frames(split: str, label: str) -> int:
        return sum(1 for r in manifest_records if r["split"] == split and r["label"] == label)
        
    stats = {
        "dataset_name": "Authentix Face Preprocessed Dataset",
        "generated_at": datetime.now().isoformat(),
        "random_seed": config.random_seed,
        "split_ratios": {
            "train": config.split.train,
            "validation": config.split.val,
            "test": config.split.test
        },
        "video_counts": {
            "train": {
                "real": len(real_train),
                "fake": len(fake_train),
                "total": len(real_train) + len(fake_train)
            },
            "validation": {
                "real": len(real_val),
                "fake": len(fake_val),
                "total": len(real_val) + len(fake_val)
            },
            "test": {
                "real": len(real_test),
                "fake": len(fake_test),
                "total": len(real_test) + len(fake_test)
            },
            "overall": {
                "real": len(real_videos),
                "fake": len(fake_videos),
                "total": len(real_videos) + len(fake_videos)
            }
        },
        "frame_counts": {
            "train": {
                "real": count_frames("train", "real"),
                "fake": count_frames("train", "fake"),
                "total": count_frames("train", "real") + count_frames("train", "fake")
            },
            "validation": {
                "real": count_frames("validation", "real"),
                "fake": count_frames("validation", "fake"),
                "total": count_frames("validation", "real") + count_frames("validation", "fake")
            },
            "test": {
                "real": count_frames("test", "real"),
                "fake": count_frames("test", "fake"),
                "total": count_frames("test", "real") + count_frames("test", "fake")
            },
            "overall": {
                "real": sum(1 for r in manifest_records if r["label"] == "real"),
                "fake": sum(1 for r in manifest_records if r["label"] == "fake"),
                "total": len(manifest_records)
            }
        }
    }
    
    stats_json_file = dataset_root / "dataset_statistics.json"
    try:
        with open(stats_json_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
        logging.info(f"Saved dataset statistics JSON to {stats_json_file}")
    except Exception as e:
        logging.error(f"Failed to save statistics JSON: {e}")
        
    # 9. Generate Summary Report (dataset_summary.md)
    elapsed_time = time.time() - start_time
    summary_report = f"""# Authentix Dataset Summary Report
Generated At: {stats["generated_at"]}
Random Seed: {config.random_seed}
Execution Duration: {elapsed_time:.2f} seconds

## Video Distribution
| Split | Real Videos | Fake Videos | Total Videos |
| --- | --- | --- | --- |
| Train ({config.split.train * 100:.1f}%) | {stats["video_counts"]["train"]["real"]} | {stats["video_counts"]["train"]["fake"]} | {stats["video_counts"]["train"]["total"]} |
| Validation ({config.split.val * 100:.1f}%) | {stats["video_counts"]["validation"]["real"]} | {stats["video_counts"]["validation"]["fake"]} | {stats["video_counts"]["validation"]["total"]} |
| Test ({config.split.test * 100:.1f}%) | {stats["video_counts"]["test"]["real"]} | {stats["video_counts"]["test"]["fake"]} | {stats["video_counts"]["test"]["total"]} |
| **Overall** | **{stats["video_counts"]["overall"]["real"]}** | **{stats["video_counts"]["overall"]["fake"]}** | **{stats["video_counts"]["overall"]["total"]}** |

## Frame Counts (Cropped Faces)
| Split | Real Frames | Fake Frames | Total Frames |
| --- | --- | --- | --- |
| Train | {stats["frame_counts"]["train"]["real"]} | {stats["frame_counts"]["train"]["fake"]} | {stats["frame_counts"]["train"]["total"]} |
| Validation | {stats["frame_counts"]["validation"]["real"]} | {stats["frame_counts"]["validation"]["fake"]} | {stats["frame_counts"]["validation"]["total"]} |
| Test | {stats["frame_counts"]["test"]["real"]} | {stats["frame_counts"]["test"]["fake"]} | {stats["frame_counts"]["test"]["total"]} |
| **Overall** | **{stats["frame_counts"]["overall"]["real"]}** | **{stats["frame_counts"]["overall"]["fake"]}** | **{stats["frame_counts"]["overall"]["total"]}** |
"""
    
    summary_md_file = dataset_root / "dataset_summary.md"
    try:
        with open(summary_md_file, "w", encoding="utf-8") as f:
            f.write(summary_report)
        logging.info(f"Saved dataset summary markdown to {summary_md_file}")
    except Exception as e:
        logging.error(f"Failed to save summary report: {e}")
        
    logging.info(f"Dataset building completed successfully in {elapsed_time:.2f} seconds.")
    return stats


if __name__ == "__main__":
    app_config = get_default_config()
    setup_logging(app_config)
    logging.info("Starting dry-run verification of dataset_builder.py")
    
    build_split_dataset(app_config)
