"""
predictor.py

This module contains the primary public API for the Authentix Multimodal Fusion Engine.
It exposes the predict_multimodal function and the Predictor class to integrate
the scoring logic and report generators into a single function call.
"""

import logging
from pathlib import Path
import sys
from typing import Optional, Tuple, Dict, Any

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.fusion.config import FusionConfig, get_default_config, setup_directories, setup_logging
from ai.fusion.schemas import VideoPrediction, AudioPrediction, LipSyncPrediction, FusionResult
from ai.fusion.fusion_engine import FusionEngine
from ai.fusion.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class Predictor:
    """
    Public wrapper interface facilitating unified multimodal prediction and reporting.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        """
        Initializes the Predictor with config parameters, generating directory targets
        and setting up logging paths.
        """
        self.config = config or get_default_config()
        
        # Initialize directories and logging
        setup_directories(self.config)
        setup_logging(self.config)
        
        self.engine = FusionEngine(self.config)
        self.report_generator = ReportGenerator(self.config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def predict(
        self,
        video: VideoPrediction,
        audio: AudioPrediction,
        lip_sync: LipSyncPrediction,
        reference_id: Optional[str] = None
    ) -> FusionResult:
        """
        Executes decision fusion and optionally writes JSON, Markdown, and HTML report assets.
        
        Args:
            video (VideoPrediction): Video deepfake prediction data.
            audio (AudioPrediction): Audio deepfake prediction data.
            lip_sync (LipSyncPrediction): Lip sync anomaly analysis data.
            reference_id (Optional[str]): Target identifier. If supplied, reports are saved to disk.
            
        Returns:
            FusionResult: Consolidated assessment object.
        """
        self.logger.info("Executing multimodal deepfake predictor pipeline...")
        
        # 1. Perform scoring and risk classification
        result = self.engine.fuse(video, audio, lip_sync)
        
        # 2. Generate report assets if a reference_id is provided
        if reference_id is not None:
            self.logger.info(f"Generating assessment report files for reference ID: {reference_id}")
            report_paths = self.report_generator.generate_reports(result, reference_id)
            
            # Append generated paths to result execution stats
            new_stats = dict(result.execution_stats)
            new_stats["reports"] = {k: str(v) for k, v in report_paths.items()}
            
            # Re-instantiate FusionResult containing updated stats
            result = FusionResult(
                prediction=result.prediction,
                confidence=result.confidence,
                video_score=result.video_score,
                audio_score=result.audio_score,
                lip_sync_score=result.lip_sync_score,
                fusion_score=result.fusion_score,
                risk_level=result.risk_level,
                reasoning=result.reasoning,
                timestamp=result.timestamp,
                execution_stats=new_stats
            )
            
        return result


def predict_multimodal(
    video: VideoPrediction,
    audio: AudioPrediction,
    lip_sync: LipSyncPrediction,
    config: Optional[FusionConfig] = None,
    reference_id: Optional[str] = None
) -> FusionResult:
    """
    Standard entrypoint helper to execute multimodal fusion predictions in a single function call.
    
    Args:
        video (VideoPrediction): Video classifier outputs.
        audio (AudioPrediction): Audio classifier outputs.
        lip_sync (LipSyncPrediction): Lip sync match classifier outputs.
        config (Optional[FusionConfig]): Custom config.
        reference_id (Optional[str]): Target reference ID to trigger report output saves.
        
    Returns:
        FusionResult: Aggregated decision assessment.
    """
    predictor = Predictor(config)
    return predictor.predict(video, audio, lip_sync, reference_id=reference_id)


if __name__ == "__main__":
    print("Executing self-test for fusion/predictor.py...")
    import tempfile
    
    # Setup mock inputs
    v_input = VideoPrediction(is_fake=True, score=0.94, confidence=0.90)
    a_input = AudioPrediction(is_fake=True, score=0.91, confidence=0.88)
    l_input = LipSyncPrediction(is_fake=False, score=0.18, confidence=0.95)
    
    try:
        # Resolve temp config to verify with disk writes
        temp_dir = Path(tempfile.mkdtemp())
        default_cfg = get_default_config()
        
        from ai.fusion.config import ReportConfig
        temp_report_cfg = ReportConfig(output_dir=temp_dir)
        temp_cfg = FusionConfig(
            project_root=default_cfg.project_root,
            fusion_root=default_cfg.fusion_root,
            weights=default_cfg.weights,
            thresholds=default_cfg.thresholds,
            risk_levels=default_cfg.risk_levels,
            logging=default_cfg.logging,
            reports=temp_report_cfg
        )
        
        # Test predict_multimodal with report generation
        ref = "self_test_run"
        res = predict_multimodal(v_input, a_input, l_input, config=temp_cfg, reference_id=ref)
        
        print(f"Aggregated Result Prediction: {res.prediction}")
        print(f"Aggregated Result Risk: {res.risk_level}")
        print(f"Generated reasoning statements count: {len(res.reasoning)}")
        print(f"Execution report stats: {res.execution_stats}")
        
        assert res.prediction == "Deepfake"
        assert res.risk_level == "High"
        assert len(res.reasoning) > 0
        assert "reports" in res.execution_stats
        assert Path(res.execution_stats["reports"]["json"]).exists()
        
        # Cleanup
        for path_str in res.execution_stats["reports"].values():
            p = Path(path_str)
            if p.exists():
                p.unlink()
        Path(res.execution_stats["reports"]["json"]).parent.rmdir()
        temp_dir.rmdir()
        
        print("All self-tests completed successfully.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
