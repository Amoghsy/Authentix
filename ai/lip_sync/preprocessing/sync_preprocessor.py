"""
sync_preprocessor.py

This module contains the SyncPreprocessor class, which coordinates the end-to-end
preprocessing pipeline. It ingests video, loads and resamples frames to 25 FPS,
extracts 16 kHz mono WAV audio, tracks the active speaker's face bounding boxes,
extracts 3D facial landmarks, crops mouth ROIs, converts audio to 100 Hz MFCC
coefficients, and aligns visual and auditory inputs into synchronized sliding-window
input tensors for the SyncNet model.
"""

import json
import logging
from pathlib import Path
import sys
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.lip_sync.preprocessing.config import ProjectConfig, get_default_config, setup_logging
from ai.lip_sync.preprocessing.video_loader import VideoLoader
from ai.lip_sync.preprocessing.audio_extractor import AudioExtractor
from ai.lip_sync.preprocessing.face_detector import FaceDetector
from ai.lip_sync.preprocessing.landmark_detector import LandmarkDetector
from ai.lip_sync.preprocessing.lip_cropper import LipCropper


class SyncPreprocessorError(Exception):
    """Raised when preprocessing pipeline orchestration fails."""
    pass


class SyncPreprocessor:
    """
    End-to-end pipeline orchestrator for audio-visual lip sync preprocessing.
    """

    def __init__(self, config: Optional[ProjectConfig] = None):
        """
        Initializes the pipeline preprocessor and its constituent modules.
        """
        self.config = config or get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Instantiate sub-modules
        self.video_loader = VideoLoader(self.config)
        self.audio_extractor = AudioExtractor(self.config)
        self.face_detector = FaceDetector(self.config)
        self.landmark_detector = LandmarkDetector(self.config)
        self.lip_cropper = LipCropper(self.config)
        
        # Audio MFCC transformer using torchaudio
        # Setup MFCC params: 400 n_fft (25ms), 160 hop (10ms) -> 100Hz feature rate
        self.mfcc_transform = T.MFCC(
            sample_rate=self.config.audio.sample_rate,
            n_mfcc=self.config.audio.num_mfcc,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 26,
                "center": True  # center=True ensures 100Hz frame rate aligns with samples count
            }
        )

    def extract_audio_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        Computes 13-dimensional Mel-Frequency Cepstral Coefficients (MFCCs) at 100 Hz.
        
        Args:
            waveform (np.ndarray): Peak-normalized mono float waveform array.
            
        Returns:
            np.ndarray: MFCC coefficients matrix of shape (num_mfcc, num_frames).
        """
        waveform_tensor = torch.from_numpy(waveform).float()
        
        # If waveform tensor lacks batch dimension, add it: [1, num_samples]
        if len(waveform_tensor.shape) == 1:
            waveform_tensor = waveform_tensor.unsqueeze(0)
            
        with torch.no_grad():
            # mfcc shape: [batch=1, num_mfcc=13, num_frames]
            mfcc_tensor = self.mfcc_transform(waveform_tensor)
            
        # Squeeze batch dimension and return numpy array
        return mfcc_tensor.squeeze(0).numpy()

    def align_visual_and_audio(
        self,
        crops: List[np.ndarray],
        mfcc: np.ndarray,
        video_fps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Groups visual frames and audio features into aligned 0.2s sliding windows.
        
        Visual segments: 5 frames (representing 0.2s at 25 fps).
        Audio segments: 20 MFCC frames (representing 0.2s at 100 Hz).
        Each video frame corresponds to 4 audio frames.
        
        Args:
            crops (List[np.ndarray]): List of normalized grayscale mouth crops (size 111x111).
            mfcc (np.ndarray): Audio MFCC matrix of shape (13, num_audio_frames).
            video_fps (float): Video sampling frame rate (must be 25.0).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Video tensor: shape (Batch, 1, 5, 111, 111)
                - Audio tensor: shape (Batch, 1, 20, 13)
        """
        num_video_frames = len(crops)
        num_audio_frames = mfcc.shape[1]
        
        # 1 video frame = (mfcc_fps / video_fps) audio frames = 100 / 25 = 4 audio frames
        audio_frames_per_video_frame = int(round(self.config.audio.mfcc_fps / video_fps))
        
        video_seq_len = self.config.lip.sequence_length  # 5 frames
        audio_seq_len = video_seq_len * audio_frames_per_video_frame  # 20 frames
        
        video_segments = []
        audio_segments = []
        
        # Slide windows across frames
        # For frame index t, the segment starts at t and covers video_seq_len frames.
        # Audio segment starts at t * audio_frames_per_video_frame and covers audio_seq_len frames.
        max_t = num_video_frames - video_seq_len
        
        for t in range(max_t + 1):
            # Resolve audio boundary indices
            audio_start = t * audio_frames_per_video_frame
            audio_end = audio_start + audio_seq_len
            
            # Ensure we do not index out of bounds in audio stream
            if audio_end > num_audio_frames:
                break
                
            # Extract video segment (shape: 5, 111, 111)
            v_seg = np.stack(crops[t:t + video_seq_len], axis=0)
            
            # Extract audio segment (shape: 13, 20) -> Transpose to (20, 13) for SyncNet conv layer
            a_seg = mfcc[:, audio_start:audio_end].T
            
            video_segments.append(v_seg)
            audio_segments.append(a_seg)
            
        if len(video_segments) == 0:
            raise SyncPreprocessorError(
                f"Insufficient temporal length to build a single aligned sequence. "
                f"Video frames: {num_video_frames}, Audio frames: {num_audio_frames}"
            )
            
        # Convert segments list to tensors
        # Video: [Batch, 5, 111, 111] -> Insert channel dim -> [Batch, 1, 5, 111, 111]
        video_tensor = torch.from_numpy(np.stack(video_segments, axis=0)).float().unsqueeze(1)
        # Audio: [Batch, 20, 13] -> Insert channel dim -> [Batch, 1, 20, 13]
        audio_tensor = torch.from_numpy(np.stack(audio_segments, axis=0)).float().unsqueeze(1)
        
        return video_tensor, audio_tensor

    def process(self, video_path: Path, output_subdir_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Orchestrates full lip sync preprocessing pipeline on an input video file.
        
        Args:
            video_path (Path): Path to raw input video file.
            output_subdir_name (Optional[str]): Target name for output dataset directory. If None,
                                                resolves to video name.
                                                
        Returns:
            Dict[str, Any]: Execution report containing output paths, frame counts, and metadata.
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Source video file not found: {video_path}")
            
        ref_name = output_subdir_name or video_path.stem
        output_dir = self.config.paths.dataset_dir / "processed" / ref_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"========== Starting Lip Sync Preprocessing: {video_path.name} ==========")
        
        # 1. Load Video Frames (sampled to config target FPS, e.g. 25 FPS)
        target_fps = self.config.video.target_fps
        self.logger.info(f"Step 1: Extracting and resampling video frames to {target_fps} FPS...")
        try:
            frames, video_metadata = self.video_loader.extract_frames(video_path, target_fps=target_fps)
            self.logger.info(f"Extracted {len(frames)} frames. Original FPS: {video_metadata.fps:.2f}")
        except Exception as e:
            raise SyncPreprocessorError(f"Video extraction failed: {e}")
            
        # 2. Extract and Normalize Audio
        self.logger.info("Step 2: Demuxing and peak-normalizing audio track...")
        output_wav = output_dir / "audio.wav"
        try:
            self.audio_extractor.extract_audio(video_path, output_wav)
            waveform, sample_rate = self.audio_extractor.load_waveform(output_wav)
            self.logger.info(f"Audio extracted successfully: samples={len(waveform)}, sample_rate={sample_rate}")
        except Exception as e:
            raise SyncPreprocessorError(f"Audio extraction failed: {e}")
            
        # 3. Detect and Track Face
        self.logger.info("Step 3: Detecting and tracking active speaker face...")
        try:
            detections = self.face_detector.detect_faces(frames, batch_size=4)
            face_track = self.face_detector.track_face(detections)
            self.logger.info("Face tracking complete.")
        except Exception as e:
            raise SyncPreprocessorError(f"Face detection or tracking failed: {e}")
            
        # 4. Extract Facial Landmarks
        self.logger.info("Step 4: Extracting facial landmarks...")
        landmarks_seq = []
        for i, (frame, bbox) in enumerate(zip(frames, face_track)):
            if bbox is not None:
                lm = self.landmark_detector.detect_landmarks(frame, bbox)
            else:
                lm = self.landmark_detector.detect_landmarks(frame)
            landmarks_seq.append(lm)
            
        # Check if landmarks were successfully extracted at least somewhere
        valid_lms_count = sum(1 for lm in landmarks_seq if lm is not None)
        self.logger.info(f"Landmarks extracted: {valid_lms_count}/{len(frames)} frames valid.")
        
        # 5. Crop Mouth ROIs
        self.logger.info("Step 5: Cropping mouth regions of interest...")
        try:
            crops = self.lip_cropper.crop_sequence(frames, landmarks_seq)
            crops_dir = output_dir / "mouth_crops"
            self.lip_cropper.save_crops(crops, crops_dir)
        except Exception as e:
            raise SyncPreprocessorError(f"Mouth ROI cropping failed: {e}")
            
        # 6. Extract Audio MFCC features
        self.logger.info("Step 6: Extracting audio MFCC features...")
        try:
            mfcc_feats = self.extract_audio_features(waveform)
            self.logger.info(f"MFCC extraction complete. Feature map shape: {mfcc_feats.shape}")
        except Exception as e:
            raise SyncPreprocessorError(f"Audio MFCC extraction failed: {e}")
            
        # 7. Segment and Align Tensors
        self.logger.info("Step 7: Aligning video frames and audio MFCC segments...")
        try:
            video_tensor, audio_tensor = self.align_visual_and_audio(crops, mfcc_feats, target_fps)
            self.logger.info(f"Alignment complete. Video tensor: {video_tensor.shape}, Audio tensor: {audio_tensor.shape}")
        except Exception as e:
            raise SyncPreprocessorError(f"Tensors alignment failed: {e}")
            
        # 8. Save final tensors and metadata
        self.logger.info("Step 8: Writing final model-ready tensors to disk...")
        video_tensor_path = output_dir / "video_input.pt"
        audio_tensor_path = output_dir / "audio_input.pt"
        
        torch.save(video_tensor, video_tensor_path)
        torch.save(audio_tensor, audio_tensor_path)
        
        metadata = {
            "reference": ref_name,
            "video_path": str(video_path),
            "video_metadata": {
                "fps": video_metadata.fps,
                "width": video_metadata.width,
                "height": video_metadata.height,
                "duration": video_metadata.duration,
                "total_frames": video_metadata.total_frames
            },
            "output_metadata": {
                "resampled_fps": target_fps,
                "processed_video_frames": len(frames),
                "processed_audio_samples": len(waveform),
                "aligned_segments_count": video_tensor.size(0),
                "video_tensor_shape": list(video_tensor.shape),
                "audio_tensor_shape": list(audio_tensor.shape)
            }
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        self.logger.info(f"Lip Sync Preprocessing completed successfully for: {video_path.name}")
        self.logger.info(f"Data saved under: {output_dir}")
        
        return {
            "output_dir": output_dir,
            "video_tensor_path": video_tensor_path,
            "audio_tensor_path": audio_tensor_path,
            "metadata_path": metadata_path,
            "metadata": metadata
        }


if __name__ == "__main__":
    # Self-test block using synthetic audio-visual assets
    print("Executing self-test for sync_preprocessor.py...")
    import tempfile
    import subprocess
    import cv2
    
    # 1. Initialize logging
    cfg = get_default_config()
    setup_logging(cfg)
    
    # 2. Build preprocessor
    try:
        preprocessor = SyncPreprocessor(cfg)
    except Exception as e:
        print(f"Failed to initialize SyncPreprocessor: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Monkeypatch detectors to bypass real face detection on synthetic drawing
    def mock_detect_faces(frames, *args, **kwargs):
        return [[((80, 80, 240, 200), 0.99)] for _ in frames]
        
    def mock_track_face(detections, *args, **kwargs):
        return [(80, 80, 240, 200) for _ in detections]
        
    def mock_detect_landmarks(frame, bbox=None):
        mesh = np.zeros((468, 3))
        # Ensure lip landmarks have spatial bounds (e.g. 0.45 to 0.55)
        for idx in preprocessor.landmark_detector.get_lip_indices():
            mesh[idx] = [np.random.uniform(0.45, 0.55), np.random.uniform(0.55, 0.65), 0.0]
        return mesh
        
    preprocessor.face_detector.detect_faces = mock_detect_faces
    preprocessor.face_detector.track_face = mock_track_face
    preprocessor.landmark_detector.detect_landmarks = mock_detect_landmarks

    # 3. Create a synthetic video with moving facial features and synchronized audio
    temp_dir = Path(tempfile.mkdtemp())
    temp_video_path = temp_dir / "synthetic_talk.mp4"
    temp_wav_path = temp_dir / "temp_audio.wav"
    
    # Params
    width, height = 320, 240
    fps = 25.0
    duration = 2.2  # 55 frames, enough for sliding windows
    num_frames = int(fps * duration)
    sample_rate = 16000
    
    print("Generating synthetic visual talking face frames...")
    # Create OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    
    try:
        # Loop to draw a face moving its mouth
        for i in range(num_frames):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            # Head (oval)
            cv2.ellipse(img, (160, 120), (60, 90), 0, 0, 360, (120, 120, 120), -1)
            # Eyes
            cv2.circle(img, (140, 100), 8, (255, 255, 255), -1)
            cv2.circle(img, (180, 100), 8, (255, 255, 255), -1)
            
            # Mouth (moves back and forth vertically to simulate talk)
            mouth_y_open = int(10 * np.sin(2 * np.pi * 3.0 * (i / fps)))
            cv2.ellipse(img, (160, 160), (25, 5 + abs(mouth_y_open)), 0, 0, 360, (0, 0, 255), -1)
            
            out.write(img)
        out.release()
        
        print("Generating synthetic synchronized audio track...")
        t_arr = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_signal = 0.4 * np.sin(2 * np.pi * 300.0 * t_arr)
        torchaudio.save(str(temp_wav_path), torch.from_numpy(audio_signal).unsqueeze(0).float(), sample_rate)
        
        # Merge audio and video streams using local FFmpeg
        print("Merging streams using FFmpeg...")
        merged_video_path = temp_dir / "synthetic_talk_merged.mp4"
        cmd = [
            preprocessor.audio_extractor.ffmpeg_cmd,
            "-y",
            "-i", str(temp_video_path),
            "-i", str(temp_wav_path),
            "-c:v", "mpeg4",
            "-c:a", "aac",
            "-shortest",
            str(merged_video_path)
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # 4. Run preprocessing pipeline
        print("Running SyncPreprocessor pipeline on synthetic video...")
        report = preprocessor.process(merged_video_path, output_subdir_name="synthetic_test_run")
        
        # 5. Assertions on output structures
        print("Verifying pipeline outputs...")
        assert report["video_tensor_path"].exists()
        assert report["audio_tensor_path"].exists()
        assert report["metadata_path"].exists()
        
        video_tensor = torch.load(report["video_tensor_path"])
        audio_tensor = torch.load(report["audio_tensor_path"])
        
        print(f"Video tensor shape: {video_tensor.shape}. Expected: [B, 1, 5, 111, 111]")
        print(f"Audio tensor shape: {audio_tensor.shape}. Expected: [B, 1, 20, 13]")
        
        assert video_tensor.ndim == 5
        assert video_tensor.size(1) == 1
        assert video_tensor.size(2) == 5
        assert video_tensor.size(3) == 111
        assert video_tensor.size(4) == 111
        
        assert audio_tensor.ndim == 4
        assert audio_tensor.size(1) == 1
        assert audio_tensor.size(2) == 20
        assert audio_tensor.size(3) == 13
        
        assert video_tensor.size(0) == audio_tensor.size(0)
        
        print(f"Metadata aligned segments count: {report['metadata']['output_metadata']['aligned_segments_count']}")
        print("All sync_preprocessor.py self-tests: PASSED")
        
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
        
    finally:
        # Cleanup
        for p in [temp_video_path, temp_wav_path, merged_video_path]:
            if p.exists():
                p.unlink()
        try:
            temp_dir.rmdir()
        except OSError:
            pass
