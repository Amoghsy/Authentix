"""Helper runner script to preprocess a sampled subset of the audio dataset.

This script selects a specific number of real and fake files from the raw dataset,
copies them to a temporary folder, runs the end-to-end preprocessing orchestrator,
and cleans up the temporary files afterwards.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.audio.preprocessing.config import AudioPreprocessingConfig
from ai.audio.preprocessing.run_preprocessing import PreprocessingPipeline


def run_sampled_pipeline(limit_per_class: int, clear_resume: bool) -> None:
    """Copies a limited number of real/fake files, runs the pipeline, and cleans up.

    Args:
        limit_per_class: Max number of files to process per class label.
        clear_resume: Clear existing report progress and processed folder.
    """
    config = AudioPreprocessingConfig()
    
    # 1. Locate raw files
    source_raw = config.raw_dir
    if not source_raw.exists():
        print(f"Error: Raw dataset folder not found at {source_raw}")
        sys.exit(1)
        
    print(f"Scanning raw files from {source_raw}...")
    
    real_files = []
    fake_files = []
    
    # Recursively find source audio files
    for p in source_raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in config.audio_extensions:
            parts = [part.lower() for part in p.parts]
            if "real" in parts:
                real_files.append(p)
            elif "fake" in parts:
                fake_files.append(p)
                
    print(f"Discovered overall raw dataset: {len(real_files)} real files, {len(fake_files)} fake files.")
    
    if len(real_files) < limit_per_class or len(fake_files) < limit_per_class:
        print(
            f"Warning: Requested {limit_per_class} per class, but only found "
            f"{len(real_files)} real files and {len(fake_files)} fake files. "
            f"Reducing limit to the maximum available count."
        )
        limit_per_class = min(len(real_files), len(fake_files))
        
    # Sample files
    sampled_real = real_files[:limit_per_class]
    sampled_fake = fake_files[:limit_per_class]
    
    # 2. Setup temporary staging directory
    temp_raw_dir = config.dataset_root / "temp_sampled_raw"
    if temp_raw_dir.exists():
        shutil.rmtree(temp_raw_dir)
        
    # Create matching labels structure
    (temp_raw_dir / "real").mkdir(parents=True, exist_ok=True)
    (temp_raw_dir / "fake").mkdir(parents=True, exist_ok=True)
    
    print(f"\nStaging {limit_per_class} real files and {limit_per_class} fake files into {temp_raw_dir}...")
    
    for f in sampled_real:
        shutil.copy2(f, temp_raw_dir / "real" / f.name)
    for f in sampled_fake:
        shutil.copy2(f, temp_raw_dir / "fake" / f.name)
        
    print("Staging complete. Running preprocessing orchestrator...")
    
    # 3. Create overridden configuration pointing to temp_raw_dir
    overridden_config = AudioPreprocessingConfig(
        raw_dir=temp_raw_dir,
        num_workers=config.num_workers
    )
    
    pipeline = PreprocessingPipeline(overridden_config)
    
    # Run the pipeline
    success = False
    try:
        success = pipeline.run(clear_resume=clear_resume)
    finally:
        # 4. Cleanup temp directory
        print("\nCleaning up staged temporary files...")
        if temp_raw_dir.exists():
            shutil.rmtree(temp_raw_dir)
            
    if success:
        print("\nPre-processing for the sampled 2,000 files completed successfully.")
        sys.exit(0)
    else:
        print("\nPipeline preprocessing failed.")
        sys.exit(1)


def main() -> None:
    """CLI Entrypoint for sampled run."""
    parser = argparse.ArgumentParser(description="Run Authentix Audio pipeline on a sampled class subset.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of files to process per class label (real and fake). Default 1000."
    )
    parser.add_argument(
        "--clear-resume",
        action="store_true",
        help="Clear existing report progress and processed folder for a clean run."
    )
    args = parser.parse_args()
    
    run_sampled_pipeline(args.limit, args.clear_resume)


if __name__ == "__main__":
    main()
