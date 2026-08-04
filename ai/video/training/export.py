"""
export.py

This module contains the model export script for Authentix. It loads PyTorch weights,
converts the DeepfakeModel to ONNX format with dynamic batch-size axes, verifies the graph's
mathematical equivalence against PyTorch using ONNX Runtime, and runs speed benchmarks.
"""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional

# Force stdout/stderr to use UTF-8 to prevent CP1252 encoding crashes on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging
from ai.video.training.model import DeepfakeModel


def export_to_onnx(
    config: AppConfig, 
    checkpoint_path: Optional[Path] = None, 
    output_path: Optional[Path] = None,
    is_dry_run: bool = False
) -> Path:
    """
    Exports PyTorch model weights to ONNX format and verifies accuracy & latency.
    
    Args:
        config (AppConfig): Configuration settings.
        checkpoint_path (Optional[Path]): Input PyTorch checkpoint file (.pth).
        output_path (Optional[Path]): Target output ONNX filepath.
        is_dry_run (bool): If True, exports a newly initialized random model.
        
    Returns:
        Path: Output ONNX filepath.
    """
    # 1. Resolve paths
    if checkpoint_path is None:
        checkpoint_path = config.paths.checkpoints_dir / "best_model.pth"
        
    if output_path is None:
        output_path = config.paths.exports_dir / "deepfake_model.onnx"
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info("Initializing model for ONNX export...")
    model = DeepfakeModel(dropout=0.0, pretrained=False)
    
    if not is_dry_run:
        logging.info(f"Loading weights from checkpoint: {checkpoint_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logging.info("Dry-run: Exporting a random-weights model.")
        
    model.eval()
    
    # 2. Export process
    # Create dummy input with configuration size
    height, width = config.pipeline.image_size
    dummy_input = torch.randn(1, 3, height, width, requires_grad=False)
    
    logging.info(f"Exporting model to ONNX: {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14, # Stable opset supporting MobileNetV3
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    logging.info("ONNX export complete.")
    
    # 3. ONNX Structure Verification
    logging.info("Verifying ONNX graph structure integrity...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logging.info("ONNX model structure checked successfully.")
    
    # 4. Mathematical Equivalence Verification via ONNX Runtime
    logging.info("Verifying ONNX model numerical equivalence against PyTorch...")
    ort_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    
    # Generate test input
    test_input = torch.randn(2, 3, height, width) # Batch size of 2 to test dynamic axes
    
    # PyTorch inference
    with torch.no_grad():
        py_output = model(test_input).numpy()
        
    # ONNX inference
    ort_inputs = {"input": test_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_output = ort_outs[0]
    
    # Check max differences
    difference = np.max(np.abs(py_output - onnx_output))
    logging.info(f"Maximum discrepancy between PyTorch and ONNX logits: {difference:.2e}")
    
    # Assert tolerance limit (1e-4)
    tolerance = 1e-4
    if difference > tolerance:
        raise ValueError(f"ONNX discrepancy {difference} exceeds tolerance {tolerance}!")
    logging.info("✅ Mathematical equivalence check passed.")
    
    # 5. Latency Benchmarking (PyTorch CPU vs ONNX CPU)
    logging.info("Benchmarking CPU inference speed (50 runs)...")
    np_input = test_input.numpy()
    
    # PyTorch warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model(test_input)
            
    # PyTorch runs
    t_py_start = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = model(test_input)
    t_py = (time.time() - t_py_start) / 50.0
    
    # ONNX warm-up
    for _ in range(10):
        _ = ort_session.run(None, ort_inputs)
        
    # ONNX runs
    t_onnx_start = time.time()
    for _ in range(50):
        _ = ort_session.run(None, ort_inputs)
    t_onnx = (time.time() - t_onnx_start) / 50.0
    
    speedup = t_py / t_onnx if t_onnx > 0 else 1.0
    logging.info(
        f"Latency Summary (Batch Size 2) -\n"
        f"  PyTorch CPU: {t_py*1000.0:.2f} ms / step\n"
        f"  ONNX CPU:    {t_onnx*1000.0:.2f} ms / step\n"
        f"  ONNX Speedup: {speedup:.2f}x faster than PyTorch CPU."
    )
    
    return output_path


def main() -> None:
    """
    CLI main entrypoint.
    """
    default_config = get_default_config()
    
    parser = argparse.ArgumentParser(
        description="Authentix - Export trained PyTorch weights to ONNX format"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to trained best_model.pth. Defaults to config location"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Target filepath for exported .onnx file"
    )
    
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a quick end-to-end self-test/dry-run using limited mock data."
    )
    
    args = parser.parse_args()
    
    if args.self_test:
        logging.info("Starting dry-run verification of export.py")
        mock_onnx_path = default_config.paths.exports_dir / "test_deepfake_model.onnx"
        try:
            export_to_onnx(
                config=default_config,
                checkpoint_path=None,
                output_path=mock_onnx_path,
                is_dry_run=True
            )
            if mock_onnx_path.exists():
                mock_onnx_path.unlink()
                logging.info("Cleaned up mock ONNX file successfully.")
            logging.info("export.py module verified successfully.")
            sys.exit(0)
        except Exception as e:
            logging.error(f"export.py verification failed: {e}")
            sys.exit(1)
            
    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    out_path = Path(args.output_path) if args.output_path else None
    
    setup_logging(default_config)
    
    try:
        export_to_onnx(default_config, ckpt_path, out_path, is_dry_run=False)
    except Exception as e:
        logging.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
