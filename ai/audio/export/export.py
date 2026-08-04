"""Main entry point to execute the Audio AI Model Export Pipeline.

This script parses options, initializes config states, sets up loggers,
instantiates the model exporter to serialize weights to ONNX,
and runs the verifier to ensure outputs align within precision tolerances.
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path to enable absolute imports when run directly
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ai.audio.export.config import ExportConfig, setup_logger
from ai.audio.export.exporter import AudioModelExporter
from ai.audio.export.verifier import AudioONNXVerifier


def run_export_pipeline() -> None:
    """Orchestrates model configuration, ONNX tracing, and correctness verification."""
    # 1. Initialize Configuration
    config = ExportConfig()
    try:
        config.validate()
    except Exception as e:
        print(f"Export configuration validation failed: {e}")
        sys.exit(1)
        
    config.create_directories()

    # 2. Setup Logging
    logger = setup_logger("model_export", config.logging.log_file, config.logging.log_level)
    logger.info("=== Starting Authentix Audio Model Export & Optimization Pipeline ===")

    # 3. Initialize Exporter and Serialize PyTorch to ONNX
    try:
        exporter = AudioModelExporter(config)
        export_report = exporter.export_to_onnx()
    except Exception as e:
        logger.error(f"ONNX Model Export process execution failed: {e}")
        sys.exit(1)

    # 4. Load PyTorch model for verification mapping checks
    try:
        pytorch_model = exporter.load_model()
    except Exception as e:
        logger.error(f"Failed to load PyTorch model weights for verification check: {e}")
        sys.exit(1)

    # 5. Run Correctness Verification
    try:
        verifier = AudioONNXVerifier(config)
        verification_report = verifier.verify_onnx_model(pytorch_model)
    except Exception as e:
        logger.error(f"ONNX verification check execution failed: {e}")
        sys.exit(1)

    # 6. Print Export Completion Summary Dashboard
    print_export_dashboard(export_report, verification_report)
    logger.info("=== Audio Model Export Pipeline Completed Successfully ===")


def print_export_dashboard(export_report: dict, verification_report: dict) -> None:
    """Prints a formatted console dashboard detailing export metrics and results.

    Args:
        export_report: Export performance stats dict.
        verification_report: Verification discrepancy stats dict.
    """
    test_cases_info = ""
    for case in verification_report["test_cases"]:
        test_cases_info += (
            f"  - Case {case['test_case']} [Shape: {case['batch_size']}x{case['sequence_length']}]:\n"
            f"      Status:            {case['status']}\n"
            f"      Max Abs Error:     {case['max_absolute_error']:.4e}\n"
            f"      Max Rel Error:     {case['max_relative_error']:.4e}\n"
            f"      Predictions Match: {case['predictions_equal']}\n"
        )

    dashboard = f"""
================================================================================
AUTHENTIX AUDIO MODEL EXPORT SUMMARY DASHBOARD
================================================================================
Export Pipeline Status:         {export_report['status']}
Export Processing Time:        {export_report['export_duration_seconds']:.2f} seconds
Serialized Model File Size:     {export_report['file_size_mb']:.2f} MB
Exported Model Path:           {export_report['onnx_model_path']}
--------------------------------------------------------------------------------
Verification Compliance Match:  {verification_report['status']}
Absolute Precision Tolerance:   {verification_report['absolute_tolerance']:.1e}
Relative Precision Tolerance:   {verification_report['relative_tolerance']:.1e}
--------------------------------------------------------------------------------
Dynamic Axis Test Runs:
{test_cases_info}================================================================================
"""
    print(dashboard)


def main() -> None:
    """CLI Entrypoint for audio export pipeline execution."""
    run_export_pipeline()


if __name__ == "__main__":
    main()
