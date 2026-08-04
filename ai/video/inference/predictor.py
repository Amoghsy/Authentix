"""
predictor.py

This module contains the DeepfakePredictor class, which serves as the inference engine
for video deepfake face crops. It supports loading both PyTorch (.pth) checkpoints
and optimized ONNX (.onnx) runtime formats, processes single images, batches, or folders,
and returns labels and confidence probabilities.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Any, List, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# Force stdout/stderr to use UTF-8 to prevent CP1252 encoding crashes on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Insert project root to sys.path to allow executing the script directly
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.video.preprocessing.config import AppConfig, get_default_config, setup_logging
from ai.video.training.augmentations import get_val_transforms
from ai.video.training.model import DeepfakeModel


class DeepfakePredictor:
    """
    Unified Inference Engine for Authentix Deepfake Detection.
    Handles image loading, preprocessing, model execution, and prediction mapping.
    """
    def __init__(
        self,
        model_path: Path,
        config: AppConfig,
        device: str = "cpu"
    ) -> None:
        """
        Args:
            model_path (Path): Path to .pth weights file or .onnx graph model.
            config (AppConfig): Root application configuration.
            device (str): Inference device ('cpu' or 'cuda').
        """
        self.model_path = Path(model_path)
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        
        self.label_mapping = {0: "real", 1: "fake"}
        self.transform = get_val_transforms(config.pipeline.image_size)
        
        # Identify model format and load
        if self.model_path.suffix.lower() == ".onnx":
            self.mode = "onnx"
            import onnxruntime as ort
            logging.info(f"Loading ONNX model for inference from: {self.model_path}")
            # Automatically choose CPU/CUDA providers for ONNX Runtime
            providers = ["CPUExecutionProvider"]
            if self.device.type == "cuda":
                providers.insert(0, "CUDAExecutionProvider")
            self.ort_session = ort.InferenceSession(str(self.model_path), providers=providers)
            logging.info(f"ONNX model loaded successfully on providers: {self.ort_session.get_providers()}")
        elif self.model_path.suffix.lower() in {".pth", ".pt"}:
            self.mode = "pytorch"
            logging.info(f"Loading PyTorch model for inference from: {self.model_path}")
            self.model = DeepfakeModel(dropout=0.0, pretrained=False)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            logging.info("PyTorch model loaded and set to eval mode successfully.")
        else:
            raise ValueError(f"Unsupported model extension: {self.model_path.suffix}. Must be .pth, .pt, or .onnx")

    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Loads and preprocesses a single image.
        
        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, 3, H, W).
        """
        # Load image in RGB
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Apply transforms (val pipeline uses resize & normalize)
        augmented = self.transform(image=image_np)
        img_tensor = augmented["image"] # Shape (3, H, W)
        
        # Convert to numpy and add batch dimension (1, 3, H, W)
        return img_tensor.unsqueeze(0).numpy()

    def predict_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Predicts label and probability for a single image crop.
        
        Args:
            image_path (Path): Path to input crop.
            
        Returns:
            Dict[str, Any]: Prediction dictionary containing label, confidence score, and probabilities.
        """
        try:
            input_data = self._preprocess_image(image_path)
        except Exception as e:
            logging.error(f"Failed to preprocess image at {image_path}: {e}")
            return {
                "label": "unknown",
                "confidence": 0.0,
                "probabilities": {"real": 0.5, "fake": 0.5},
                "error": str(e)
            }
            
        # Run inference
        if self.mode == "onnx":
            ort_inputs = {self.ort_session.get_inputs()[0].name: input_data}
            ort_outs = self.ort_session.run(None, ort_inputs)
            logits = ort_outs[0] # numpy array shape (1, 2)
            
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs = probs[0] # shape (2,)
        else:
            input_tensor = torch.from_numpy(input_data).to(self.device)
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
        pred_idx = int(np.argmax(probs))
        pred_label = self.label_mapping[pred_idx]
        confidence = float(probs[pred_idx])
        
        return {
            "label": pred_label,
            "confidence": confidence,
            "probabilities": {
                "real": float(probs[0]),
                "fake": float(probs[1])
            }
        }

    def predict_batch(self, image_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Runs batch prediction on a list of image paths.
        
        Args:
            image_paths (List[Path]): List of absolute image paths.
            
        Returns:
            List[Dict[str, Any]]: List of prediction dictionaries.
        """
        results = []
        for path in image_paths:
            results.append(self.predict_image(path))
        return results

    def predict_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """
        Discovers all images recursively in a folder and runs predictions.
        """
        dir_path = Path(dir_path)
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [
            p for p in dir_path.rglob("*") 
            if p.suffix.lower() in image_extensions
        ]
        
        logging.info(f"Discovered {len(image_files)} images in folder: {dir_path}")
        
        predictions = []
        for img_path in image_files:
            pred = self.predict_image(img_path)
            pred["image_path"] = str(img_path.resolve())
            predictions.append(pred)
            
        return predictions


def main() -> None:
    """
    CLI interface for running deepfake prediction.
    """
    default_config = get_default_config()
    
    parser = argparse.ArgumentParser(
        description="Authentix - Deepfake Video Crop Predictor"
    )
    parser.add_argument(
        "--model-path",
        required=False,
        type=str,
        help="Path to trained PyTorch (.pth) checkpoint or ONNX (.onnx) model file."
    )
    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to a single face crop image."
    )
    parser.add_argument(
        "--dir-path",
        type=str,
        help="Path to directory containing face crops."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device ('cpu' or 'cuda')."
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a quick end-to-end self-test/dry-run using limited mock data."
    )
    
    args = parser.parse_args()
    
    setup_logging(default_config)
    
    if args.self_test:
        logging.info("Starting dry-run verification of predictor.py")
        try:
            # Create a dummy image
            dummy_img_path = default_config.paths.checkpoints_dir / "test_crop.jpg"
            dummy_img_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save a random image to disk
            dummy_data = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            Image.fromarray(dummy_data).save(dummy_img_path)
            
            # Create a dummy weights checkpoint to load (dry-run mode test)
            dummy_ckpt_path = default_config.paths.checkpoints_dir / "dummy_weights.pth"
            dummy_model = DeepfakeModel(dropout=0.0, pretrained=False)
            torch.save({
                "model_state_dict": dummy_model.state_dict()
            }, dummy_ckpt_path)
            
            # Initialize predictor on PyTorch dummy model
            predictor = DeepfakePredictor(
                model_path=dummy_ckpt_path,
                config=default_config,
                device="cpu"
            )
            
            # Predict on dummy image
            res = predictor.predict_image(dummy_img_path)
            logging.info(f"Mock Image prediction: {res}")
            
            assert "label" in res and "confidence" in res, "Prediction output structure is missing keys!"
            assert res["label"] in {"real", "fake"}, f"Unknown predicted label output: {res['label']}"
            
            # Clean up files created
            if dummy_img_path.exists():
                dummy_img_path.unlink()
            if dummy_ckpt_path.exists():
                dummy_ckpt_path.unlink()
            logging.info("Cleaned up mock files successfully.")
            logging.info("predictor.py module verified successfully.")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Predictor verification failed: {e}")
            sys.exit(1)
            
    if not args.model_path:
        parser.error("the following arguments are required: --model-path")
        
    model_path = Path(args.model_path)
    predictor = DeepfakePredictor(model_path, default_config, device=args.device)
    
    if args.image_path:
        img_path = Path(args.image_path)
        res = predictor.predict_image(img_path)
        logging.info(f"\nPrediction for {img_path.name}:")
        logging.info(f"  Class Label: {res['label'].upper()}")
        logging.info(f"  Confidence:  {res['confidence']*100:.2f}%")
        logging.info(f"  Probabilities - Real: {res['probabilities']['real']:.4f} | Fake: {res['probabilities']['fake']:.4f}")
        
    if args.dir_path:
        dir_path = Path(args.dir_path)
        preds = predictor.predict_directory(dir_path)
        for p in preds:
            logging.info(f"Image: {Path(p['image_path']).name} -> Label: {p['label'].upper()} ({p['confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()
