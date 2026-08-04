"""End-to-end pipeline orchestrator and verification runner.

This script coordinates audio file discovery, quality checks, format normalization,
dataset splitting, statistics collation, and output format verification. It logs detailed
runtime diagnostics to preprocessing.log, conversion.log, and dataset_builder.log.
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to sys.path to enable absolute imports when run directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import soundfile as sf

from ai.audio.preprocessing.audio_converter import AudioConverter, run_conversion_pool
from ai.audio.preprocessing.config import AudioPreprocessingConfig, setup_logger
from ai.audio.preprocessing.dataset_builder import DatasetBuilder


class PreprocessingPipeline:
    """Orchestrates audio preprocessing steps and executes compliance verification."""

    def __init__(self, config: AudioPreprocessingConfig):
        """Initializes the PreprocessingPipeline with configuration.

        Args:
            config: An instance of AudioPreprocessingConfig.
        """
        self.config = config
        self.config.validate()
        self.config.create_directories()

        self.logger = setup_logger(
            "preprocessing",
            self.config.logs_dir / "preprocessing.log"
        )
        self.logger.info("Initialized PreprocessingPipeline.")

    def run(self, clear_resume: bool = False) -> bool:
        """Executes the end-to-end preprocessing, partition, and verification pipeline.

        Args:
            clear_resume: If True, deletes existing progress states and does a fresh run.

        Returns:
            bool: True if pipeline completed and all verification tests passed, False otherwise.
        """
        pipeline_start_time = time.time()
        self.logger.info("=== Starting Authentix Audio Preprocessing Pipeline ===")

        if clear_resume:
            self.logger.info("Clearing previous progress states as requested.")
            if self.config.conversion_report_path.exists():
                os.remove(self.config.conversion_report_path)
            if self.config.processed_dir.exists():
                shutil.rmtree(self.config.processed_dir)
            self.config.create_directories()

        # Step 1: Discover and Convert Audio
        self.logger.info("Step 1: Audio Discovery, Quality Validation, and Format Conversion")
        converter = AudioConverter(self.config)
        raw_files = converter.discover_files()
        
        if not raw_files:
            self.logger.error("No raw audio files discovered. Pipeline aborted.")
            return False

        conversion_start = time.time()
        processed_files, skipped_files = run_conversion_pool(
            converter,
            raw_files,
            num_workers=self.config.num_workers
        )
        conversion_duration = time.time() - conversion_start

        if not processed_files:
            self.logger.error("No audio files were successfully processed. Pipeline aborted.")
            return False

        # Step 2: Build Dataset Splits and Generate Metadata
        self.logger.info("Step 2: Grouping, Splitting, and Packaging Dataset Splits")
        builder = DatasetBuilder(self.config)
        
        splits = builder.build_splits(processed_files)
        manifest_records = builder.populate_split_directories(splits)
        
        builder.generate_manifest(manifest_records)
        builder.generate_label_mapping()
        
        stats = builder.compute_and_save_statistics(manifest_records)
        builder.generate_preprocessing_report(len(processed_files), skipped_files, stats)

        # Step 3: Compliance & Quality Verification
        self.logger.info("Step 3: Verification Checks on Processed Splits")
        verification_passed = self.verify_processed_dataset(manifest_records)

        pipeline_duration = time.time() - pipeline_start_time

        # Pipeline Summary Report Logging
        self.log_pipeline_summary(
            total_discovered=len(raw_files),
            processed_count=len(processed_files),
            skipped_count=len(skipped_files),
            conversion_duration=conversion_duration,
            total_duration=pipeline_duration,
            stats=stats,
            verification_status="PASSED" if verification_passed else "FAILED"
        )

        return verification_passed

    def verify_processed_dataset(self, manifest_records: List[Dict]) -> bool:
        """Performs 100% compliance verification on the processed audio output splits.

        Verifies:
        - Files are non-empty and readable.
        - Correct sample rate (16000 Hz).
        - Correct target channels (1 - Mono).
        - Format subtype matches 16-bit PCM WAV.
        - Folder label placement matches manifest label records.

        Args:
            manifest_records: List of processed manifest records containing target paths.

        Returns:
            bool: True if every file is compliant, False if any check fails.
        """
        self.logger.info(f"Starting verification on {len(manifest_records)} dataset files.")
        failures = 0

        for record in manifest_records:
            file_path = Path(record["file_path"])
            label = record["label"]
            split = record["split"]

            # 1. Existence and non-emptiness checks
            if not file_path.exists():
                self.logger.error(f"Verification FAILED: File does not exist at {file_path}")
                failures += 1
                continue

            if file_path.stat().st_size == 0:
                self.logger.error(f"Verification FAILED: Zero-byte file found at {file_path}")
                failures += 1
                continue

            # 2. Directory structure verification
            parent_label_folder = file_path.parent.name
            parent_split_folder = file_path.parent.parent.name
            
            if parent_label_folder != label:
                self.logger.error(
                    f"Verification FAILED: Folder label '{parent_label_folder}' does not match "
                    f"manifest label '{label}' for {file_path}"
                )
                failures += 1

            if parent_split_folder != split:
                self.logger.error(
                    f"Verification FAILED: Folder split '{parent_split_folder}' does not match "
                    f"manifest split '{split}' for {file_path}"
                )
                failures += 1

            # 3. Audio format compliance checking (16kHz, mono, PCM_16 WAV)
            try:
                info = sf.info(str(file_path))
                if info.samplerate != self.config.sample_rate:
                    self.logger.error(
                        f"Verification FAILED: Sample rate is {info.samplerate} Hz, "
                        f"expected {self.config.sample_rate} Hz for {file_path}"
                    )
                    failures += 1

                if info.channels != self.config.target_channels:
                    self.logger.error(
                        f"Verification FAILED: Channels count is {info.channels}, "
                        f"expected {self.config.target_channels} (Mono) for {file_path}"
                    )
                    failures += 1

                if info.format != "WAV":
                    self.logger.error(
                        f"Verification FAILED: Format is {info.format}, "
                        f"expected WAV for {file_path}"
                    )
                    failures += 1

                if info.subtype != "PCM_16":
                    self.logger.error(
                        f"Verification FAILED: Subtype encoding is {info.subtype}, "
                        f"expected PCM_16 (16-bit PCM) for {file_path}"
                    )
                    failures += 1

            except Exception as e:
                self.logger.error(f"Verification FAILED: Unreadable metadata on {file_path}. Error: {e}")
                failures += 1

        if failures > 0:
            self.logger.error(f"Verification completed with {failures} compliance failures.")
            return False

        self.logger.info("Compliance verification PASSED for 100% of dataset split files.")
        return True

    def log_pipeline_summary(
        self,
        total_discovered: int,
        processed_count: int,
        skipped_count: int,
        conversion_duration: float,
        total_duration: float,
        stats: Dict,
        verification_status: str
    ) -> None:
        """Prints and writes the final execution statistics summary.

        Args:
            total_discovered: Discovered files count.
            processed_count: Successfully processed files count.
            skipped_count: Skipped/failed validation files count.
            conversion_duration: Audio conversion run time.
            total_duration: Overall execution duration.
            stats: Computed statistics dictionary.
            verification_status: Verification results string ("PASSED" or "FAILED").
        """
        overall = stats["overall"]
        throughput = processed_count / conversion_duration if conversion_duration > 0 else 0
        disk_space_mb = overall["total_storage_size_bytes"] / (1024 * 1024)

        summary_dashboard = f"""
================================================================================
AUTHENTIX AUDIO PIPELINE SUMMARY DASHBOARD
================================================================================
Pipeline Execution Duration:    {total_duration:.2f} seconds
Conversion Processing Speed:   {throughput:.2f} files/second (Elapsed: {conversion_duration:.2f}s)
Total Raw Audio Discovered:     {total_discovered} files
Successfully Preprocessed:      {processed_count} files
Skipped / Quality Rejections:   {skipped_count} files
--------------------------------------------------------------------------------
Dataset Size on Disk:           {disk_space_mb:.2f} MB
Aggregate Playback Duration:    {overall["total_duration_seconds"]:.2f} seconds
Dataset Label Balance Ratio:    {overall["dataset_balance_ratio"]} (Real: {overall["real_count"]} | Fake: {overall["fake_count"]})
--------------------------------------------------------------------------------
Verification Compliance Status: {verification_status}
================================================================================
"""
        print(summary_dashboard)
        self.logger.info(summary_dashboard)


def main() -> None:
    """CLI entry point for pipeline orchestration."""
    parser = argparse.ArgumentParser(description="Run Authentix Audio Preprocessing Pipeline.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        help="Custom path to raw audio files directory."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of parallel worker processes."
    )
    parser.add_argument(
        "--clear-resume",
        action="store_true",
        help="Clear existing report progress and processed folder for a clean run."
    )
    args = parser.parse_args()

    # Load defaults, apply overrides if provided
    config_overrides = {}
    if args.raw_dir:
        config_overrides["raw_dir"] = Path(args.raw_dir)
    if args.num_workers:
        config_overrides["num_workers"] = args.num_workers

    config = AudioPreprocessingConfig(**config_overrides)
    pipeline = PreprocessingPipeline(config)
    success = pipeline.run(clear_resume=args.clear_resume)

    if success:
        print("Pipeline execution completed successfully.")
        os._exit(0)
    else:
        print("Pipeline execution failed. Check logs for details.")
        os._exit(1)


if __name__ == "__main__":
    main()
