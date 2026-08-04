"""Audio validation, format conversion, and normalization engine.

This module is responsible for discovering all raw audio files, validating their quality
(handling zero-byte, corrupted, clipped, extremely short, or duplicate files), resampling
to 16 kHz, downmixing to mono, performing peak normalization, and writing the clean output
to 16-bit PCM WAV files in the processed target directory. It supports multiprocessing and
resuming from interrupted runs.
"""

import hashlib
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

# Add project root to sys.path to enable absolute imports when run directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.audio.preprocessing.config import AudioPreprocessingConfig, setup_logger


@dataclass
class AudioFileInfo:
    """Metadata container for a processed audio file."""

    source_path: str
    target_path: str
    duration_seconds: float
    sample_rate: int
    channels: int
    label: str
    file_hash: str
    file_size_bytes: int


class AudioConverter:
    """Discovers, validates, normalizes, and converts raw audio files to standard format."""

    def __init__(self, config: AudioPreprocessingConfig):
        """Initializes the AudioConverter with configuration.

        Args:
            config: An instance of AudioPreprocessingConfig.
        """
        self.config = config
        self.config.validate()
        self.config.create_directories()

        # Initialize dedicated logger
        self.logger = setup_logger(
            "audio_converter",
            self.config.logs_dir / "conversion.log"
        )
        self.logger.info("Initialized AudioConverter.")

        # Reports state
        self.processed_files: Dict[str, Dict] = {}
        self.skipped_files: Dict[str, str] = {}  # source_path -> skip_reason
        self.seen_hashes: Set[str] = set()
        self.seen_filenames: Set[str] = set()

        self._load_resume_state()

    def _load_resume_state(self) -> None:
        """Loads previous conversion report to support resumable runs."""
        report_path = self.config.conversion_report_path
        if report_path.exists():
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = json.load(f)
                
                # Restore processed files
                self.processed_files = report.get("processed_files", {})
                self.skipped_files = report.get("skipped_files", {})
                
                # Rebuild seen sets to prevent duplicates
                for details in self.processed_files.values():
                    self.seen_hashes.add(details.get("file_hash", ""))
                    target_path = Path(details.get("target_path", ""))
                    self.seen_filenames.add(target_path.name.lower())
                
                self.logger.info(
                    f"Resuming pipeline. Loaded {len(self.processed_files)} processed files "
                    f"and {len(self.skipped_files)} skipped files from existing report."
                )
            except Exception as e:
                self.logger.warning(f"Failed to load resume state: {e}. Starting fresh.")

    def _save_resume_state(self) -> None:
        """Saves current processed and skipped file status to conversion_report.json."""
        report_path = self.config.conversion_report_path
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processed_count": len(self.processed_files),
            "skipped_count": len(self.skipped_files),
            "processed_files": self.processed_files,
            "skipped_files": self.skipped_files,
        }
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save conversion report: {e}")

    def discover_files(self) -> List[Path]:
        """Recursively discovers all audio files matching supported extensions in raw_dir.

        Returns:
            List[Path]: A list of paths to discovered audio files.
        """
        self.logger.info(f"Discovering audio files in {self.config.raw_dir}")
        all_files = []
        if not self.config.raw_dir.exists():
            self.logger.error(f"Raw directory does not exist: {self.config.raw_dir}")
            return []

        for p in self.config.raw_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.config.audio_extensions:
                all_files.append(p)

        self.logger.info(f"Discovered {len(all_files)} audio files with target extensions.")
        return sorted(all_files)

    @staticmethod
    def extract_label(file_path: Path) -> str:
        """Extracts the label ('real' or 'fake') from the directory structure of the file.

        Args:
            file_path: Path of the audio file.

        Returns:
            str: Label string 'real', 'fake', or 'unknown'.
        """
        parts = [p.lower() for p in file_path.parts]
        if "real" in parts:
            return "real"
        if "fake" in parts:
            return "fake"
        return "unknown"

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Computes SHA-256 hash of a file to check for content duplicates.

        Args:
            file_path: Path of the file.

        Returns:
            str: Hex digest of SHA-256 hash.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def validate_audio(self, file_path: Path) -> Tuple[bool, str, Optional[Tuple[torch.Tensor, int]]]:
        """Validates audio file quality, format, and content metrics.

        Checks:
        1. Zero-byte check.
        2. Format readability and header health.
        3. Extremely short audio file duration.
        4. Digital clipping check.

        Args:
            file_path: Path of the file to check.

        Returns:
            Tuple[bool, str, Optional[Tuple[Tensor, int]]]:
                - bool: True if valid, False if rejected.
                - str: Rejection reason or "OK".
                - Optional[Tuple[Tensor, int]]: Waveform tensor and original sample rate if valid.
        """
        # 1. Zero-byte check
        try:
            if file_path.stat().st_size == 0:
                return False, "Zero-length file", None
        except Exception as e:
            return False, f"Unreadable file stats: {e}", None

        # 2. Header and readability check
        try:
            # We try using torchaudio first, falling back to soundfile if needed
            try:
                info = torchaudio.info(str(file_path))
                sr = info.sample_rate
                channels = info.num_channels
                frames = info.num_frames
                duration = frames / sr if sr > 0 else 0
            except Exception:
                # Fallback to soundfile metadata extraction
                info_sf = sf.info(str(file_path))
                sr = info_sf.samplerate
                channels = info_sf.channels
                duration = info_sf.duration

            if duration < self.config.min_duration_seconds:
                return False, f"Extremely short recording ({duration:.3f}s)", None

            # Load actual audio waveform to check clipping
            try:
                waveform, load_sr = torchaudio.load(str(file_path))
            except Exception:
                # Fallback load using soundfile
                data, load_sr = sf.read(str(file_path), dtype="float32")
                if len(data.shape) == 1:
                    data = np.expand_dims(data, axis=0)
                else:
                    data = data.T  # (channels, frames)
                waveform = torch.from_numpy(data)

        except Exception as e:
            return False, f"Unreadable/Corrupted format: {str(e)}", None

        # 3. Clipping check (runs of consecutive peak samples)
        if self._check_clipping(waveform, threshold=self.config.clipping_threshold):
            return False, "Clipped audio (consecutive peak samples detected)", None

        return True, "OK", (waveform, load_sr)

    @staticmethod
    def _check_clipping(waveform: torch.Tensor, threshold: float = 0.999, consecutive_limit: int = 5) -> bool:
        """Detects digital clipping by searching for flat-topped peaks.

        Args:
            waveform: Audio waveform tensor.
            threshold: Amplitude threshold to check.
            consecutive_limit: Number of consecutive samples at/above threshold to define clipping.

        Returns:
            bool: True if clipping is detected, False otherwise.
        """
        for channel in waveform:
            at_peak = torch.abs(channel) >= threshold
            if consecutive_limit <= 1:
                if torch.any(at_peak):
                    return True
            else:
                # Use numpy convolution to find consecutive peak runs
                peak_runs = np.convolve(at_peak.numpy().astype(int), np.ones(consecutive_limit, dtype=int), mode="valid")
                if np.any(peak_runs >= consecutive_limit):
                    return True
        return False

    def convert_and_save(
        self,
        waveform: torch.Tensor,
        orig_sr: int,
        source_path: Path,
        label: str,
        file_hash: str
    ) -> AudioFileInfo:
        """Processes (resamples, downmixes, normalizes) and exports file as 16-bit PCM WAV.

        Args:
            waveform: Loaded audio waveform tensor.
            orig_sr: Original sample rate.
            source_path: Path to the original file.
            label: The extracted label.
            file_hash: Precomputed SHA-256 hash.

        Returns:
            AudioFileInfo: Metadata container containing output details.
        """
        # Determine the target path inside processed directory
        # Maintain label subfolder: processed/real/filename.wav or processed/fake/filename.wav
        target_name = f"{source_path.stem}.wav"
        target_path = self.config.processed_dir / label / target_name
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Channel conversion to Mono (average multi-channel)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Resampling to target sample rate
        if orig_sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.config.sample_rate)
            waveform = resampler(waveform)

        # 3. Peak Normalization (scale to absolute peak of 0.9 to avoid digital distortion)
        max_amplitude = torch.max(torch.abs(waveform)).item()
        if max_amplitude > 0:
            waveform = (waveform / max_amplitude) * 0.9

        # Convert back to numpy array for soundfile write
        np_waveform = waveform.numpy().squeeze()

        # 4. Write out to 16-bit PCM WAV using Soundfile
        sf.write(
            str(target_path),
            np_waveform,
            self.config.sample_rate,
            subtype="PCM_16"
        )

        duration = waveform.shape[1] / self.config.sample_rate
        file_size = target_path.stat().st_size

        return AudioFileInfo(
            source_path=str(source_path.resolve()),
            target_path=str(target_path.resolve()),
            duration_seconds=duration,
            sample_rate=self.config.sample_rate,
            channels=1,
            label=label,
            file_hash=file_hash,
            file_size_bytes=file_size
        )

    def process_single_file(self, file_path: Path) -> Tuple[str, str, Optional[Dict]]:
        """Validates and processes a single audio file.

        Helper method designed to run inside multiprocessing workers.

        Args:
            file_path: Path of the audio file to process.

        Returns:
            Tuple[str, str, Optional[Dict]]:
                - str: Source path string.
                - str: Status code/reason ('SUCCESS' or skip reason).
                - Optional[Dict]: Serialized AudioFileInfo dict if SUCCESS.
        """
        # Extract label early
        label = self.extract_label(file_path)
        if label == "unknown":
            return str(file_path), "Unknown label directory", None

        # Compute content hash
        try:
            file_hash = self.compute_sha256(file_path)
        except Exception as e:
            return str(file_path), f"Hash error: {e}", None

        # Run validations
        is_valid, reason, load_data = self.validate_audio(file_path)
        if not is_valid or load_data is None:
            return str(file_path), reason, None

        waveform, orig_sr = load_data

        try:
            info = self.convert_and_save(waveform, orig_sr, file_path, label, file_hash)
            return str(file_path), "SUCCESS", asdict(info)
        except Exception as e:
            return str(file_path), f"Conversion failure: {str(e)}", None


def run_conversion_pool(
    converter: AudioConverter,
    files: List[Path],
    num_workers: int
) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """Orchestrates multi-processed audio validation and conversion.

    Args:
        converter: Initialized AudioConverter instance.
        files: List of file Paths to process.
        num_workers: Number of concurrent process workers.

    Returns:
        Tuple[Dict[str, Dict], Dict[str, str]]:
            - Dict of successfully processed files.
            - Dict of skipped files mapped to their skip reasons.
    """
    to_process = []
    
    # Filter files that are already completed in resume state
    for f in files:
        f_str = str(f.resolve())
        # Check if already processed
        if f_str in converter.processed_files:
            # Confirm file still exists
            tgt = Path(converter.processed_files[f_str]["target_path"])
            if tgt.exists():
                continue
        # Check if already skipped
        if f_str in converter.skipped_files:
            continue

        # Prevent duplicate filenames or payload hashes discovered during sequential check
        f_name_lower = f.name.lower()
        if f_name_lower in converter.seen_filenames:
            converter.skipped_files[f_str] = f"Duplicate filename: '{f.name}'"
            continue
        converter.seen_filenames.add(f_name_lower)

        try:
            f_hash = converter.compute_sha256(f)
            if f_hash in converter.seen_hashes:
                converter.skipped_files[f_str] = "Duplicate file payload (hash collision)"
                continue
            converter.seen_hashes.add(f_hash)
        except Exception as e:
            converter.skipped_files[f_str] = f"Hash error: {e}"
            continue

        to_process.append(f)

    if not to_process:
        converter.logger.info("No new files to process. Processed/Skipped directories are up to date.")
        return converter.processed_files, converter.skipped_files

    converter.logger.info(f"Submitting {len(to_process)} audio files for parallel processing.")

    # Multiprocessing execution pool
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(converter.process_single_file, f): f for f in to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting Audio"):
            f_path = futures[future]
            try:
                src, status, metadata = future.result()
                if status == "SUCCESS" and metadata is not None:
                    converter.processed_files[src] = metadata
                else:
                    converter.skipped_files[src] = status
                    converter.logger.warning(f"Skipped {f_path.name}: {status}")
            except Exception as e:
                src_str = str(f_path.resolve())
                converter.skipped_files[src_str] = f"Process error: {e}"
                converter.logger.error(f"Process error on {f_path.name}: {e}")

    elapsed = time.time() - start_time
    throughput = len(to_process) / elapsed if elapsed > 0 else 0
    converter.logger.info(
        f"Parallel processing completed. Duration: {elapsed:.2f}s | Throughput: {throughput:.2f} files/sec"
    )

    # Save state
    converter._save_resume_state()

    return converter.processed_files, converter.skipped_files
