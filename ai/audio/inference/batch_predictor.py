"""Batch predictor execution module for standalone audio inference.

This module recursively scans directories for audio files, runs ONNX predictions
on each sample, computes execution statistics, and exports tabular predictions.csv
and metadata summary.json reports.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path to enable absolute imports when run directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.audio.inference.config import InferenceConfig
from ai.audio.inference.predictor import AudioPredictor


class AudioBatchPredictor:
    """Recursively processes folders of audio recordings, running batched ONNX predictions."""

    def __init__(self, config: InferenceConfig):
        """Initializes the AudioBatchPredictor.

        Args:
            config: InferenceConfig instance.
        """
        self.config = config
        self.logger = logging.getLogger("audio_inference.batch_predictor")
        
        self.config.validate()
        self.config.create_directories()

        self.predictor = AudioPredictor(self.config)
        
        # Audio extensions supported by torchaudio and soundfile fallbacks
        self.audio_extensions: Set[str] = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

    def predict_folder(
        self,
        folder_path: Union[str, Path],
        recursive: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """Recursively scans and processes folders of audio recordings.

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

        self.logger.info(f"Scanning audio recordings in {folder_path.resolve()}")
        
        # 1. Discover all matching audio files
        audio_files = []
        glob_pattern = "**/*" if recursive else "*"
        for p in folder_path.glob(glob_pattern):
            if p.is_file() and p.suffix.lower() in self.audio_extensions:
                audio_files.append(p)

        self.logger.info(f"Discovered {len(audio_files)} files for evaluation.")
        if len(audio_files) == 0:
            self.logger.warning("No audio files discovered matching supported extensions.")
            return pd.DataFrame(), {}

        # 2. Run inference loop
        results = []
        latencies = []
        failures = 0

        progress_bar = tqdm(
            audio_files,
            desc="Batch Processing Audio",
            unit="file",
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
                    "latency_ms": res["latency_ms"],
                    "status": "SUCCESS"
                }
                latencies.append(res["latency_ms"])
            except Exception as e:
                self.logger.error(f"Failed to process file {file_path.name}: {e}")
                failures += 1
                result_row = {
                    "file_path": str(file_path.resolve()),
                    "file_name": file_path.name,
                    "prediction": "Unknown",
                    "confidence": 0.0,
                    "fake_probability": 0.0,
                    "real_probability": 0.0,
                    "latency_ms": 0.0,
                    "status": f"FAILED: {e}"
                }
            
            results.append(result_row)
            progress_bar.set_postfix({"failed": failures})

        # 3. Construct DataFrame
        predictions_df = pd.DataFrame(results)

        # 4. Calculate batch summary statistics
        total_files = len(audio_files)
        successful_runs = total_files - failures

        if successful_runs > 0:
            avg_latency = float(np.mean(latencies))
            real_count = int(np.sum(predictions_df["prediction"] == "Real"))
            fake_count = int(np.sum(predictions_df["prediction"] == "Fake"))
            real_percent = round((real_count / successful_runs) * 100.0, 2)
            fake_percent = round((fake_count / successful_runs) * 100.0, 2)
        else:
            avg_latency = 0.0
            real_count = fake_count = 0
            real_percent = fake_percent = 0.0

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": total_files,
            "successful_predictions": successful_runs,
            "failed_predictions": failures,
            "real_classifications": real_count,
            "fake_classifications": fake_count,
            "real_percentage": real_percent,
            "fake_percentage": fake_percent,
            "average_latency_ms": round(avg_latency, 2),
            "status": "SUCCESS" if failures == 0 else "PARTIAL_SUCCESS"
        }

        # 5. Export results to disk
        self._save_outputs(predictions_df, summary)

        return predictions_df, summary

    def _save_outputs(self, predictions_df: pd.DataFrame, summary: Dict) -> None:
        """Saves batch predictions CSV and summary JSON to the results folder."""
        output_dir = self.config.output_dir

        predictions_path = output_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"Saved batch predictions CSV to {predictions_path.name}")

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        self.logger.info(f"Saved batch summary JSON to {summary_path.name}")


def main() -> None:
    """CLI Entrypoint for running batch folder predictions."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Run batch audio deepfake folder predictions.")
    parser.add_argument("folder_path", type=str, help="Path to the audio files folder.")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive scanning of subdirectories."
    )
    args = parser.parse_args()

    config = InferenceConfig()
    batch_predictor = AudioBatchPredictor(config)
    
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
