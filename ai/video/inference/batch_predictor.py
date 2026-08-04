"""Batch predictor execution module for video deepfake detection.

This module recursively scans directories for video files, runs ONNX predictions
on each sample, computes performance metrics (latency, FPS), and exports tabular
predictions.csv and metadata summary.json reports.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
# Configure path references to enable direct execution
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.video.inference.config import InferenceConfig
from ai.video.inference.predictor import VideoPredictor


class VideoBatchPredictor:
    """Recursively processes folders of video recordings, running batched ONNX predictions."""

    def __init__(self, config: InferenceConfig):
        """Initializes the VideoBatchPredictor.

        Args:
            config: InferenceConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("video_inference.batch_predictor")

        self.config.validate()
        self.config.create_directories()

        self.predictor = VideoPredictor(self.config)
        
        # Supported video file formats
        self.video_extensions: Set[str] = {".mp4", ".avi", ".mov", ".mkv"}

    def predict_folder(
        self,
        folder_path: Union[str, Path],
        recursive: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """Recursively scans and processes folders of video files.

        Args:
            folder_path: Path of the folder to scan.
            recursive: If True, scans all subdirectories recursively.

        Returns:
            Tuple[pd.DataFrame, Dict]: Containing:
                - predictions_df: Detailed prediction rows.
                - summary_report: Aggregated run statistics.
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        self.logger.info(f"Scanning video files in: {folder_path.resolve()}")
        
        # 1. Discover all matching video files
        video_files = []
        glob_pattern = "**/*" if recursive else "*"
        for p in folder_path.glob(glob_pattern):
            if p.is_file() and p.suffix.lower() in self.video_extensions:
                video_files.append(p)

        self.logger.info(f"Discovered {len(video_files)} videos for evaluation.")
        if len(video_files) == 0:
            self.logger.warning("No video files discovered matching supported extensions.")
            return pd.DataFrame(), {}

        # 2. Run inference loop
        results = []
        latencies = []
        total_frames = 0
        failures = 0
        
        start_batch_time = time.perf_counter()

        progress_bar = tqdm(
            video_files,
            desc="Batch Processing Videos",
            unit="video",
            leave=True
        )

        for file_path in progress_bar:
            try:
                res = self.predictor.predict(file_path)
                result_row = {
                    "file_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "prediction": res["prediction"],
                    "confidence": res["confidence"],
                    "fake_probability": res["fake_probability"],
                    "real_probability": res["real_probability"],
                    "frames_processed": res["frames_processed"],
                    "latency_ms": res["latency_ms"],
                    "status": "SUCCESS"
                }
                latencies.append(res["latency_ms"])
                total_frames += res["frames_processed"]
            except Exception as e:
                self.logger.error(f"Failed to process video {file_path.name}: {e}")
                failures += 1
                result_row = {
                    "file_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "prediction": "Unknown",
                    "confidence": 0.0,
                    "fake_probability": 0.0,
                    "real_probability": 0.0,
                    "frames_processed": 0,
                    "latency_ms": 0.0,
                    "status": f"FAILED: {e}"
                }
            
            results.append(result_row)
            progress_bar.set_postfix({"failed": failures})

        total_batch_duration = time.perf_counter() - start_batch_time

        # 3. Construct DataFrame
        predictions_df = pd.DataFrame(results)

        # 4. Calculate batch summary statistics
        total_files = len(video_files)
        successful_runs = total_files - failures

        if successful_runs > 0:
            avg_latency = float(np.mean(latencies))
            real_count = int(np.sum(predictions_df["prediction"] == "Real"))
            fake_count = int(np.sum(predictions_df["prediction"] == "Fake"))
            real_percent = round((real_count / successful_runs) * 100.0, 2)
            fake_percent = round((fake_count / successful_runs) * 100.0, 2)
            
            # FPS = frames processed / elapsed seconds
            avg_fps = total_frames / total_batch_duration if total_batch_duration > 0 else 0.0
        else:
            avg_latency = 0.0
            real_count = fake_count = 0
            real_percent = fake_percent = 0.0
            avg_fps = 0.0

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": total_files,
            "successful_predictions": successful_runs,
            "failed_predictions": failures,
            "real_classifications": real_count,
            "fake_classifications": fake_count,
            "real_percentage": real_percent,
            "fake_percentage": fake_percent,
            "total_frames_processed": total_frames,
            "total_duration_seconds": round(total_batch_duration, 2),
            "average_latency_ms": round(avg_latency, 2),
            "average_fps": round(avg_fps, 2),
            "status": "SUCCESS" if failures == 0 else "PARTIAL_SUCCESS"
        }

        # 5. Export results to disk
        self._save_outputs(predictions_df, summary)

        return predictions_df, summary

    def _save_outputs(self, predictions_df: pd.DataFrame, summary: Dict) -> None:
        """Saves batch predictions CSV and summary JSON to the results folder."""
        output_dir = self.config.output_dir

        predictions_path = output_dir / self.config.predictions_csv
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"Saved batch predictions CSV to {predictions_path.name}")

        summary_path = output_dir / self.config.summary_json
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        self.logger.info(f"Saved batch summary JSON to {summary_path.name}")


def main() -> None:
    """CLI Entrypoint for running batch folder predictions."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Run batch video deepfake folder predictions.")
    parser.add_argument("folder_path", type=str, help="Path to the video files folder.")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive scanning of subdirectories."
    )
    args = parser.parse_args()

    config = InferenceConfig()
    batch_predictor = VideoBatchPredictor(config)
    
    try:
        _, summary = batch_predictor.predict_folder(
            folder_path=args.folder_path,
            recursive=not args.no_recursive
        )
        print(json.dumps(summary, indent=4))
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
