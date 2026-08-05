"""
dependencies.py

This module manages singleton dependencies for the Authentix FastAPI backend.
It lazily instantiates and caches VideoPredictor, AudioPredictor, SyncPreprocessor,
and FusionEngine instances, reusing them across request sessions to prevent high reloading latency.
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

logger = logging.getLogger("backend.dependencies")

# Singleton caching variables
_video_predictor = None
_audio_predictor = None
_sync_preprocessor = None
_fusion_predictor = None


def get_video_predictor():
    """
    Returns a cached or newly initialized VideoPredictor instance.
    Reuses the underlying ONNX Runtime session.
    """
    global _video_predictor
    if _video_predictor is None:
        from ai.video.inference.config import InferenceConfig as VideoConfig
        from ai.video.inference.predictor import VideoPredictor
        logger.info("Initializing singleton VideoPredictor ONNX session...")
        settings = get_settings()
        video_cfg = VideoConfig(model_path=settings.VIDEO_MODEL_PATH)
        _video_predictor = VideoPredictor(video_cfg)
    return _video_predictor


def get_audio_predictor():
    """
    Returns a cached or newly initialized AudioPredictor instance.
    Reuses the underlying ONNX Runtime session.
    """
    global _audio_predictor
    if _audio_predictor is None:
        from ai.audio.inference.config import InferenceConfig as AudioConfig
        from ai.audio.inference.predictor import AudioPredictor
        logger.info("Initializing singleton AudioPredictor ONNX session...")
        settings = get_settings()
        audio_cfg = AudioConfig(model_path=settings.AUDIO_MODEL_PATH)
        _audio_predictor = AudioPredictor(audio_cfg)
    return _audio_predictor


def get_sync_preprocessor():
    """
    Returns a cached or newly initialized SyncPreprocessor instance.
    """
    global _sync_preprocessor
    if _sync_preprocessor is None:
        from ai.lip_sync.preprocessing.sync_preprocessor import SyncPreprocessor
        from ai.lip_sync.preprocessing.config import get_default_config as get_lipsync_config
        logger.info("Initializing singleton SyncPreprocessor instance...")
        lipsync_cfg = get_lipsync_config()
        _sync_preprocessor = SyncPreprocessor(lipsync_cfg)
    return _sync_preprocessor


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
    Eagerly instantiates all AI and Fusion singletons on server start
    to warm up ONNX Runtime sessions and prevent first-request latency spikes.
    """
    logger.info("Warming up backend AI model sessions...")
    settings = get_settings()
    
    # Check video model existence
    if not settings.VIDEO_MODEL_PATH.exists():
        logger.warning(f"Video model file missing at {settings.VIDEO_MODEL_PATH}. Skipping video predictor warm-up.")
    else:
        try:
            get_video_predictor()
        except Exception as e:
            logger.error(f"Failed to warm up video predictor: {e}")

    # Check audio model existence
    if not settings.AUDIO_MODEL_PATH.exists():
        logger.warning(f"Audio model file missing at {settings.AUDIO_MODEL_PATH}. Skipping audio predictor warm-up.")
    else:
        try:
            get_audio_predictor()
        except Exception as e:
            logger.error(f"Failed to warm up audio predictor: {e}")

    # Check face landmarker task file existence to avoid blocking downloads on startup
    if not settings.FACE_LANDMARKER_PATH.exists():
        logger.warning(f"Face landmarker task file missing at {settings.FACE_LANDMARKER_PATH}. Skipping sync preprocessor warm-up.")
    else:
        try:
            get_sync_preprocessor()
        except Exception as e:
            logger.error(f"Failed to warm up sync preprocessor: {e}")

    # Initialize fusion predictor
    try:
        get_fusion_predictor()
    except Exception as e:
        logger.error(f"Failed to warm up fusion predictor: {e}")
        
    logger.info("Backend services warm-up completed.")


if __name__ == "__main__":
    print("Executing self-test for backend/app/dependencies.py...")
    # Setup test logging
    logging.basicConfig(level=logging.INFO)
    try:
        # Eager warm up and cache assertions
        warm_up_services()
        
        v1 = get_video_predictor()
        v2 = get_video_predictor()
        assert v1 is v2
        print("VideoPredictor singleton check: PASSED")
        
        a1 = get_audio_predictor()
        a2 = get_audio_predictor()
        assert a1 is a2
        print("AudioPredictor singleton check: PASSED")
        
        s1 = get_sync_preprocessor()
        s2 = get_sync_preprocessor()
        assert s1 is s2
        print("SyncPreprocessor singleton check: PASSED")
        
        f1 = get_fusion_predictor()
        f2 = get_fusion_predictor()
        assert f1 is f2
        print("FusionPredictor singleton check: PASSED")
        
        print("All self-tests completed successfully: PASSED")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
