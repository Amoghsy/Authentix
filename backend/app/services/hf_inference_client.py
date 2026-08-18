"""
hf_inference_client.py

Client for communicating with the external Hugging Face AI Inference Service.
Transmits media files over HTTP, handles timeouts, authenticates via Bearer API Key,
and maps response payloads to native Fusion Engine prediction schemas.
"""

import logging
from pathlib import Path
import sys
import time
from typing import Optional

import httpx

# Ensure project root is in path
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.app.config import Settings, get_settings
from backend.app.exceptions import ModelInferenceError
from ai.fusion.schemas import VideoPrediction, AudioPrediction, LipSyncPrediction

logger = logging.getLogger("backend.services.hf_client")


class HFInferenceClient:
    """Client for calling external Hugging Face AI Inference endpoints."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.HF_INFERENCE_URL.rstrip("/")
        self.api_key = self.settings.HF_INFERENCE_API_KEY
        self.timeout = float(self.settings.HF_INFERENCE_TIMEOUT)

    async def check_ready(self) -> dict:
        """Checks readiness status of the Hugging Face AI service."""
        url = f"{self.base_url}/ready"
        logger.info(f"Checking HF service readiness at: {url}")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url)
                if res.status_code == 200:
                    return res.json()
                logger.warning(f"HF /ready returned status code {res.status_code}")
                return {"ready": False, "status_code": res.status_code}
        except Exception as e:
            logger.warning(f"HF service readiness check failed: {e}")
            return {"ready": False, "error": str(e)}

    async def predict_all(
        self,
        media_path: Path,
        request_id: str
    ) -> tuple[VideoPrediction, AudioPrediction, LipSyncPrediction]:
        """
        Sends uploaded media file to Hugging Face AI Inference Service (`POST /predict`).
        
        Args:
            media_path (Path): Path to the saved video file.
            request_id (str): Unique analysis session ID.
            
        Returns:
            tuple[VideoPrediction, AudioPrediction, LipSyncPrediction]:
                Mapped predictions ready for the Fusion Engine.
        """
        url = f"{self.base_url}/predict"
        headers = {
            "X-Request-ID": request_id
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info(
            f"Sending media to HF AI Service: url={url} "
            f"request_id={request_id} file={media_path.name}"
        )
        start_time = time.perf_counter()

        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                with open(media_path, "rb") as f:
                    files = {"file": (media_path.name, f, "video/mp4")}
                    response = await client.post(url, files=files, headers=headers)

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if response.status_code == 401 or response.status_code == 403:
                logger.error(f"HF Service authentication failed (HTTP {response.status_code})")
                raise ModelInferenceError("AI Service authentication failed. Check API key configuration.")

            if response.status_code != 200:
                logger.error(f"HF Service returned HTTP {response.status_code}: {response.text}")
                raise ModelInferenceError(f"AI Service error (HTTP {response.status_code}): {response.text}")

            data = response.json()
            if not data.get("success", True):
                raise ModelInferenceError(f"AI Service response reported failure: {data}")

            logger.info(
                f"HF Service inference completed successfully in {latency_ms:.1f}ms "
                f"(service internal time: {data.get('processing_time_ms', 0):.1f}ms)"
            )

            # Map response JSON to native Fusion Engine dataclasses
            v_data = data.get("video", {})
            a_data = data.get("audio", {})
            l_data = data.get("lip_sync", {})

            video_pred = VideoPrediction(
                is_fake=v_data.get("is_fake", False),
                score=float(v_data.get("score", 0.0)),
                confidence=float(v_data.get("confidence", 0.0)),
                metadata=v_data.get("metadata", {})
            )

            audio_pred = AudioPrediction(
                is_fake=a_data.get("is_fake", False),
                score=float(a_data.get("score", 0.0)),
                confidence=float(a_data.get("confidence", 0.0)),
                metadata=a_data.get("metadata", {})
            )

            lipsync_pred = LipSyncPrediction(
                is_fake=l_data.get("is_fake", False),
                score=float(l_data.get("score", 0.0)),
                confidence=float(l_data.get("confidence", 0.0)),
                metadata=l_data.get("metadata", {})
            )

            return video_pred, audio_pred, lipsync_pred

        except httpx.TimeoutException:
            logger.error(f"HF AI Service request timed out after {self.timeout}s")
            raise ModelInferenceError(f"AI Service inference timed out after {self.timeout}s.")
        except httpx.RequestError as req_err:
            logger.error(f"HF AI Service connection error: {req_err}")
            raise ModelInferenceError(f"Could not connect to AI Service at {self.base_url}: {req_err}")
        except ModelInferenceError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during HF inference call: {e}", exc_info=True)
            raise ModelInferenceError(f"AI Service evaluation failed: {e}")
