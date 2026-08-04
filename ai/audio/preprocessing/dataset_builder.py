"""Dataset partitioner, leakage protector, and statistics generator.

This module splits processed audio files into train, validation, and test sets. It prevents
data leakage by grouping files by their original recording ID. It populates final target split
directories and generates manifest files, label mapping files, statistics reports, and summaries.
"""

import csv
import json
import logging
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to sys.path to enable absolute imports when run directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.audio.preprocessing.config import AudioPreprocessingConfig, setup_logger


class DatasetBuilder:
    """Partitions processed audio files, prevents data leakage, and computes statistics."""

    def __init__(self, config: AudioPreprocessingConfig):
        """Initializes the DatasetBuilder with configurations.

        Args:
            config: An instance of AudioPreprocessingConfig.
        """
        self.config = config
        self.config.validate()
        self.config.create_directories()

        self.logger = setup_logger(
            "dataset_builder",
            self.config.logs_dir / "dataset_builder.log"
        )
        self.logger.info("Initialized DatasetBuilder.")

    @staticmethod
    def get_original_recording_id(filename: str) -> str:
        """Extracts core original recording ID from filename to prevent split leakage.

        Example:
            "file1000.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav" -> "file1000"
            "file123_augmented.wav" -> "file123"

        Args:
            filename: The basename of the audio file.

        Returns:
            str: Original recording ID.
        """
        base = filename.split(".wav")[0]
        base = base.split("_")[0]
        return base

    def build_splits(self, processed_files: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """Groups files by original recording ID and partitions them without data leakage.

        Args:
            processed_files: Dict mapping source paths to processed file metadata dicts.

        Returns:
            Dict[str, List[Dict]]: Dict with keys 'train', 'validation', 'test' mapping to file details list.
        """
        self.logger.info("Starting dataset split creation.")

        # Group file metadata by original recording ID
        grouped_files: Dict[str, List[Dict]] = defaultdict(list)
        for metadata in processed_files.values():
            target_path = Path(metadata["target_path"])
            orig_id = self.get_original_recording_id(target_path.name)
            
            # Enrich metadata with original_file_id
            enriched_meta = metadata.copy()
            enriched_meta["original_file_id"] = orig_id
            grouped_files[orig_id].append(enriched_meta)

        self.logger.info(f"Grouped {len(processed_files)} files into {len(grouped_files)} unique original recordings.")

        # Determine labels for each original recording (we check the first file of the group)
        recording_ids = list(grouped_files.keys())
        recording_labels = [grouped_files[rid][0]["label"] for rid in recording_ids]

        # First split: Train vs. Temp (Val + Test)
        temp_ratio = self.config.validation_split + self.config.test_split
        
        # Guard against extremely small datasets for splitting
        if len(recording_ids) < 3:
            self.logger.warning("Dataset is too small to perform stratified split. Defaulting all files to train split.")
            splits = {"train": [], "validation": [], "test": []}
            for rid in recording_ids:
                splits["train"].extend(grouped_files[rid])
            return splits

        train_ids, temp_ids, _, temp_labels = train_test_split(
            recording_ids,
            recording_labels,
            test_size=temp_ratio,
            random_state=self.config.random_seed,
            stratify=recording_labels
        )

        # Second split: Validation vs. Test
        val_test_ratio = self.config.test_split / temp_ratio
        
        if len(temp_ids) < 2:
            self.logger.warning("Temp split is too small for stratified partition. Defaulting temp files to validation split.")
            val_ids, test_ids = temp_ids, []
        else:
            val_ids, test_ids = train_test_split(
                temp_ids,
                test_size=val_test_ratio,
                random_state=self.config.random_seed,
                stratify=temp_labels
            )

        splits = {"train": [], "validation": [], "test": []}
        
        for rid in train_ids:
            splits["train"].extend(grouped_files[rid])
        for rid in val_ids:
            splits["validation"].extend(grouped_files[rid])
        for rid in test_ids:
            splits["test"].extend(grouped_files[rid])

        self.logger.info(
            f"Partitioned recordings: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test. "
            f"Total files in splits: {len(splits['train'])} train, {len(splits['validation'])} validation, {len(splits['test'])} test."
        )

        return splits

    def populate_split_directories(self, splits: Dict[str, List[Dict]]) -> List[Dict]:
        """Copies processed audio files to their final split folders and updates paths.

        Args:
            splits: Dict of split mappings containing metadata lists.

        Returns:
            List[Dict]: Comprehensive list of final split audio details for manifest generation.
        """
        self.logger.info("Copying processed files into target split directories.")
        final_manifest_records = []

        for split_name, files in splits.items():
            for meta in files:
                src_processed_path = Path(meta["target_path"])
                label = meta["label"]
                
                # target split folder: dataset_root / split_name / label / filename.wav
                target_dir = getattr(self.config, f"{split_name}_dir") / label
                target_dir.mkdir(parents=True, exist_ok=True)
                
                target_split_path = target_dir / src_processed_path.name
                
                # Copy file to split directory
                try:
                    shutil.copy2(src_processed_path, target_split_path)
                except Exception as e:
                    self.logger.error(f"Failed to copy {src_processed_path} to {target_split_path}: {e}")
                    continue

                record = {
                    "file_path": str(target_split_path.resolve()),
                    "relative_path": str(target_split_path.relative_to(self.config.project_root)),
                    "duration_seconds": meta["duration_seconds"],
                    "sample_rate": meta["sample_rate"],
                    "channels": meta["channels"],
                    "label": label,
                    "split": split_name,
                    "original_file_id": meta["original_file_id"],
                    "audio_hash": meta["file_hash"],
                    "file_size_bytes": meta["file_size_bytes"]
                }
                final_manifest_records.append(record)

        return final_manifest_records

    def generate_manifest(self, manifest_records: List[Dict]) -> None:
        """Writes the dataset manifest file in CSV format.

        Args:
            manifest_records: List of manifest records to write.
        """
        self.logger.info(f"Writing dataset manifest to {self.config.manifest_path}")
        fieldnames = [
            "file_path", "relative_path", "duration_seconds", "sample_rate",
            "channels", "label", "split", "original_file_id", "audio_hash", "file_size_bytes"
        ]
        try:
            with open(self.config.manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in manifest_records:
                    writer.writerow(record)
            self.logger.info("Dataset manifest CSV generated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to write manifest CSV: {e}")

    def generate_label_mapping(self) -> None:
        """Generates standard label mapping file for Wav2Vec2 mapping."""
        mapping = {"real": 0, "fake": 1}
        try:
            with open(self.config.label_mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, indent=4)
            self.logger.info(f"Label mapping written to {self.config.label_mapping_path}")
        except Exception as e:
            self.logger.error(f"Failed to write label mapping: {e}")

    def compute_and_save_statistics(self, manifest_records: List[Dict]) -> Dict:
        """Computes comprehensive dataset metrics and exports statistics files.

        Args:
            manifest_records: List of all files details in final splits.

        Returns:
            Dict: Comprehensive dictionary of calculated statistics.
        """
        self.logger.info("Computing dataset statistics.")
        
        durations = [r["duration_seconds"] for r in manifest_records]
        file_sizes = [r["file_size_bytes"] for r in manifest_records]
        
        total_files = len(manifest_records)
        real_count = sum(1 for r in manifest_records if r["label"] == "real")
        fake_count = sum(1 for r in manifest_records if r["label"] == "fake")
        
        balance_ratio = real_count / fake_count if fake_count > 0 else float("inf")

        stats = {
            "overall": {
                "total_files": total_files,
                "real_count": real_count,
                "fake_count": fake_count,
                "dataset_balance_ratio": round(balance_ratio, 4),
                "total_duration_seconds": round(sum(durations), 2),
                "min_duration_seconds": round(min(durations), 4) if durations else 0,
                "max_duration_seconds": round(max(durations), 4) if durations else 0,
                "avg_duration_seconds": round(np.mean(durations), 4) if durations else 0,
                "total_storage_size_bytes": sum(file_sizes),
                "avg_sample_rate": int(np.mean([r["sample_rate"] for r in manifest_records])) if manifest_records else 0,
            },
            "splits": {}
        }

        # Calculate statistics per split
        for split in ["train", "validation", "test"]:
            split_records = [r for r in manifest_records if r["split"] == split]
            split_durations = [r["duration_seconds"] for r in split_records]
            split_sizes = [r["file_size_bytes"] for r in split_records]
            
            s_total = len(split_records)
            s_real = sum(1 for r in split_records if r["label"] == "real")
            s_fake = sum(1 for r in split_records if r["split"] == split and r["label"] == "fake")
            s_ratio = s_real / s_fake if s_fake > 0 else float("inf")

            stats["splits"][split] = {
                "total_files": s_total,
                "real_count": s_real,
                "fake_count": s_fake,
                "balance_ratio": round(s_ratio, 4),
                "total_duration_seconds": round(sum(split_durations), 2),
                "avg_duration_seconds": round(np.mean(split_durations), 4) if split_durations else 0,
                "total_storage_size_bytes": sum(split_sizes)
            }

        # Save to dataset_statistics.json
        try:
            with open(self.config.statistics_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4)
            self.logger.info("Dataset statistics JSON generated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to write statistics JSON: {e}")

        # Generate dataset_summary.md (Human-readable report)
        self.generate_summary_markdown(stats)

        return stats

    def generate_summary_markdown(self, stats: Dict) -> None:
        """Writes a formatted Markdown summary file of the dataset.

        Args:
            stats: Computed statistics dictionary.
        """
        overall = stats["overall"]
        splits = stats["splits"]
        
        md_content = f"""# Authentix Dataset Summary

This file summarizes the metrics and distribution of the processed audio dataset.

## Overall Dataset Statistics

- **Total Files**: {overall["total_files"]}
- **Real Files**: {overall["real_count"]}
- **Fake Files**: {overall["fake_count"]}
- **Dataset Balance (Real/Fake)**: {overall["dataset_balance_ratio"]}
- **Total Duration**: {overall["total_duration_seconds"]:.2f} seconds
- **Average Duration**: {overall["avg_duration_seconds"]:.4f} seconds
- **Min / Max Duration**: {overall["min_duration_seconds"]:.3f}s / {overall["max_duration_seconds"]:.3f}s
- **Target Sample Rate**: {overall["avg_sample_rate"]} Hz (Mono)
- **Total Disk Space**: {overall["total_storage_size_bytes"] / (1024 * 1024):.2f} MB

## Dataset Partition Distribution

| Partition | Total Files | Real Files | Fake Files | Balance Ratio | Total Duration (s) | Storage Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | {splits["train"]["total_files"]} | {splits["train"]["real_count"]} | {splits["train"]["fake_count"]} | {splits["train"]["balance_ratio"]} | {splits["train"]["total_duration_seconds"]:.1f} | {splits["train"]["total_storage_size_bytes"] / (1024 * 1024):.2f} |
| **Validation** | {splits["validation"]["total_files"]} | {splits["validation"]["real_count"]} | {splits["validation"]["fake_count"]} | {splits["validation"]["balance_ratio"]} | {splits["validation"]["total_duration_seconds"]:.1f} | {splits["validation"]["total_storage_size_bytes"] / (1024 * 1024):.2f} |
| **Test** | {splits["test"]["total_files"]} | {splits["test"]["real_count"]} | {splits["test"]["fake_count"]} | {splits["test"]["balance_ratio"]} | {splits["test"]["total_duration_seconds"]:.1f} | {splits["test"]["total_storage_size_bytes"] / (1024 * 1024):.2f} |

---
*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        try:
            with open(self.config.summary_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            self.logger.info("Dataset summary Markdown file generated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to write summary markdown: {e}")

    def generate_preprocessing_report(self, processed_count: int, skipped_files: Dict[str, str], stats: Dict) -> None:
        """Writes the detailed pipeline preprocessing report file.

        Args:
            processed_count: Count of successfully converted files.
            skipped_files: Dict of skipped file paths and reasons.
            stats: Computed statistics dictionary.
        """
        overall = stats["overall"]
        
        # Categorize skip reasons
        reason_counts = defaultdict(int)
        for reason in skipped_files.values():
            # Standardize reasons for readability
            if "Clipped" in reason:
                reason_counts["Clipped Signal Rejections"] += 1
            elif "short" in reason:
                reason_counts["Duration Under Minimum Limit"] += 1
            elif "Duplicate filename" in reason:
                reason_counts["Filename Collisions"] += 1
            elif "Duplicate file payload" in reason:
                reason_counts["Content Payload Duplicates"] += 1
            elif "Corrupted" in reason or "Unreadable" in reason:
                reason_counts["Corrupted / Unreadable Formats"] += 1
            else:
                reason_counts[reason] += 1

        md_content = f"""# Authentix Preprocessing Pipeline Report

This report outlines the end-to-end processing execution of the Authentix Audio Preprocessing Pipeline.

## Pipeline Conversion Summary

- **Total Discovered Files**: {processed_count + len(skipped_files)}
- **Successfully Preprocessed**: {processed_count}
- **Skipped / Rejected**: {len(skipped_files)}
- **Yield Rate**: {processed_count / (processed_count + len(skipped_files)) * 100:.2f}% if processed_count > 0 else 0%

## Validation Failure & Skip Breakdown

| Rejection Category | File Count |
| :--- | :---: |
"""
        if reason_counts:
            for reason, count in reason_counts.items():
                md_content += f"| {reason} | {count} |\n"
        else:
            md_content += "| No files were skipped | 0 |\n"

        md_content += f"""
## Final Split Distribution

- **Training Split**: {stats['splits']['train']['total_files']} files
- **Validation Split**: {stats['splits']['validation']['total_files']} files
- **Testing Split**: {stats['splits']['test']['total_files']} files

---
*Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}*
"""
        try:
            with open(self.config.preprocessing_report_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            self.logger.info("Preprocessing report markdown generated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to write preprocessing report: {e}")
