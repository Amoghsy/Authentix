"""
dependencies.py

This module manages singleton dependencies for the Authentix FastAPI backend.
It lazily instantiates and caches the HFInferenceClient and FusionEngine instances,
reusing them across request sessions to prevent high reloading latency.
"""

import logging
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import get_settings
from backend.app.services.hf_inference_client import HFInferenceClient

logger = logging.getLogger("backend.dependencies")

# Singleton caching variables
_hf_client: Optional[HFInferenceClient] = None
_fusion_predictor = None


def get_hf_client() -> HFInferenceClient:
    """
    Returns a cached HFInferenceClient instance.
    """
    global _hf_client
    if _hf_client is None:
        logger.info("Initializing singleton HFInferenceClient...")
        settings = get_settings()
        _hf_client = HFInferenceClient(settings)
    return _hf_client


def get_fusion_predictor():
    """
    Returns a cached or newly initialized Predictor instance.
    Configured to output reports directly inside the backend's reports directory.
    """
    global _fusion_predictor
    if _fusion_predictor is None:
        from ai.fusion.config import FusionConfig, ReportConfig, get_default_config as get_fusion_config
        from ai.fusion.predictor import Predictor
        logger.info("Initializing singleton Fusion Predictor wrapper...")
        settings = get_settings()
        default_fusion_cfg = get_fusion_config()
        custom_report_cfg = ReportConfig(
            output_dir=settings.REPORTS_DIR,
            json_report_name="fusion_result.json",
            md_report_name="fusion_report.md",
            html_report_name="fusion_report.html"
        )
        custom_fusion_cfg = FusionConfig(
            project_root=default_fusion_cfg.project_root,
            fusion_root=default_fusion_cfg.fusion_root,
            weights=default_fusion_cfg.weights,
            thresholds=default_fusion_cfg.thresholds,
            risk_levels=default_fusion_cfg.risk_levels,
            logging=default_fusion_cfg.logging,
            reports=custom_report_cfg
        )
        _fusion_predictor = Predictor(custom_fusion_cfg)
    return _fusion_predictor


def warm_up_services() -> None:
    """
    Eagerly instantiates Fusion Engine on server startup.
    No local AI models are loaded on Render.
    """
    logger.info("Warming up backend Fusion Engine...")
    try:
        get_fusion_predictor()
        get_hf_client()
    except Exception as e:
        logger.error(f"Failed to warm up backend services: {e}")
    logger.info("Backend services warm-up completed.")


if __name__ == "__main__":
    print("Executing self-test for backend/app/dependencies.py...")
    logging.basicConfig(level=logging.INFO)
    try:
        warm_up_services()
        client1 = get_hf_client()
        client2 = get_hf_client()
        assert client1 is client2
        print("HFInferenceClient singleton check: PASSED")

        f1 = get_fusion_predictor()
        f2 = get_fusion_predictor()
        assert f1 is f2
        print("FusionPredictor singleton check: PASSED")

        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

