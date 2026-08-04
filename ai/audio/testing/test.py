"""Main entry point to execute the Audio AI Testing & Evaluation Pipeline.

This script parses options, initializes configuration states, sets up loggers,
instantiates the dataloaders, calls the evaluator to run inference, and invokes
the report generator to save visual and tabular reports.
"""

import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

# Add project root to sys.path to enable absolute imports when run directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.audio.testing.config import TestingConfig, setup_logger
from ai.audio.testing.dataset import AudioTestDataset, collate_audio_test_batches
from ai.audio.testing.evaluator import AudioEvaluator
from ai.audio.testing.report import AudioTestReportGenerator


def run_testing_pipeline() -> None:
    """Orchestrates configuration, loading, evaluation, and report generation."""
    # 1. Initialize Configuration
    config = TestingConfig()
    try:
        config.validate()
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)
        
    config.create_directories()

    # 2. Setup Logging
    logger = setup_logger("audio_testing", config.logging.log_file, config.logging.log_level)
    logger.info("=== Starting Authentix Audio Evaluation & Testing Pipeline ===")

    # 3. Load HuggingFace Feature Extractor (Processor)
    logger.info(f"Loading processor: {config.backbone_name}")
    try:
        processor = AutoFeatureExtractor.from_pretrained(config.backbone_name)
    except Exception as e:
        logger.error(f"Failed to load HuggingFace processor: {e}")
        sys.exit(1)

    # 4. Instantiate Dataset and Dataloader
    logger.info("Initializing testing dataset partition...")
    try:
        test_dataset = AudioTestDataset(config, processor=processor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,  # Sequential execution for evaluation consistency
            collate_fn=collate_audio_test_batches,
            num_workers=config.device.dataloader_num_workers,
            pin_memory=config.device.pin_memory
        )
    except Exception as e:
        logger.error(f"Failed to load dataset loader: {e}")
        sys.exit(1)

    # 5. Load Evaluator and Run Inference
    try:
        evaluator = AudioEvaluator(config)
        metrics, predictions_df, all_logits, all_probs = evaluator.evaluate(test_loader)
    except Exception as e:
        logger.error(f"Evaluation process execution failed: {e}")
        sys.exit(1)

    # 6. Generate Performance Reports
    try:
        report_generator = AudioTestReportGenerator(config)
        
        # Prepare targets and positive class probabilities
        y_true = predictions_df["true_label"].values
        y_probs = predictions_df["probability_fake"].values
        
        report_generator.generate_all_reports(
            metrics=metrics,
            predictions_df=predictions_df,
            y_true=y_true,
            y_probs=y_probs
        )
    except Exception as e:
        logger.error(f"Failed to generate evaluation reports: {e}")
        sys.exit(1)

    # 7. Print Console Metrics Summary
    console_report = evaluator.metrics_calculator.generate_console_summary(metrics)
    print(console_report)
    logger.info("=== Audio Evaluation Pipeline Execution Completed Successfully ===")


def main() -> None:
    """CLI Entrypoint for audio testing pipeline execution."""
    run_testing_pipeline()


if __name__ == "__main__":
    main()
