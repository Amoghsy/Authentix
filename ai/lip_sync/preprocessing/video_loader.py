"""
video_loader.py

This module contains the VideoLoader and supporting components for locating, opening,
inspecting, and decoding video files (MP4, AVI, MOV, MKV). It extracts metadata
(FPS, resolution, duration) and reads video frames with configurable temporal
sampling (such as resampling to a target FPS, step-based decimation, or extracting specific index sequences).
It includes robust error boundaries for handling corrupted video files.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import List, Tuple, Optional, Set, Generator
import cv2
import numpy as np

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.lip_sync.preprocessing.config import ProjectConfig, get_default_config


class VideoLoadError(Exception):
    """Base exception class for video loading and decoding errors."""
    pass


class VideoCorruptedError(VideoLoadError):
    """Raised when a video file is corrupted, empty, or has unreadable frames."""
    pass


@dataclass(frozen=True)
class VideoMetadata:
    """
    Stores metadata parsed from a video file.
    """
    path: Path
    fps: float
    width: int
    height: int
    total_frames: int
    duration: float  # Duration in seconds


class VideoLoader:
    """
    Handles file opening, metadata extraction, frame decoding, and sampling
    for input video files.
    """

    def __init__(self, config: Optional[ProjectConfig] = None):
        """
        Initializes the VideoLoader with an optional ProjectConfig configuration.
        """
        self.config = config or get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_metadata(self, video_path: Path) -> VideoMetadata:
        """
        Reads metadata from the specified video file without loading all frames.
        
        Args:
            video_path (Path): Path to the video file.
            
        Returns:
            VideoMetadata: The extracted metadata object.
            
        Raises:
            FileNotFoundError: If the video file does not exist.
            VideoCorruptedError: If metadata extraction fails or returns invalid properties.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found at path: {video_path}")
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoCorruptedError(f"OpenCV could not open the video file: {video_path}")
            
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Simple validation check on raw metadata
            if fps <= 0 or width <= 0 or height <= 0 or total_frames <= 0:
                raise VideoCorruptedError(
                    f"Corrupted metadata values parsed from {video_path.name}: "
                    f"FPS={fps}, Width={width}, Height={height}, FrameCount={total_frames}"
                )
                
            duration = total_frames / fps
            
            return VideoMetadata(
                path=video_path,
                fps=fps,
                width=width,
                height=height,
                total_frames=total_frames,
                duration=duration
            )
        finally:
            cap.release()

    def discover_videos(self, directory: Path) -> List[Path]:
        """
        Recursively discovers all video files inside a directory matching config extensions.
        """
        supported_exts = self.config.video.supported_extensions
        found_videos: List[Path] = []
        
        if not directory.exists() or not directory.is_dir():
            self.logger.warning(f"Scan directory does not exist or is not a directory: {directory}")
            return found_videos
            
        for filepath in directory.rglob("*"):
            if filepath.is_file() and filepath.suffix.lower() in supported_exts:
                found_videos.append(filepath)
                
        return sorted(found_videos)

    def extract_frames(
        self,
        video_path: Path,
        target_fps: Optional[float] = None,
        frame_indices: Optional[List[int]] = None
    ) -> Tuple[List[np.ndarray], VideoMetadata]:
        """
        Extracts and returns decoded frames from a video based on target FPS sampling
        or custom index selections.
        
        Args:
            video_path (Path): Path to the video file.
            target_fps (Optional[float]): If provided, resamples the frame rate to match this value.
            frame_indices (Optional[List[int]]): If provided, extracts only these specific 0-indexed frames.
            
        Returns:
            Tuple[List[np.ndarray], VideoMetadata]: A list of frames (BGR NumPy arrays) and the video metadata.
            
        Raises:
            VideoCorruptedError: If frame loading fails or is interrupted by corrupt bytes.
        """
        metadata = self.load_metadata(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoCorruptedError(f"Could not open video file: {video_path}")
            
        frames: List[np.ndarray] = []
        try:
            # Mode A: Extract frames matching specific 0-based frame indices
            if frame_indices is not None:
                sorted_indices = sorted(set(frame_indices))
                index_pointer = 0
                
                for idx in sorted_indices:
                    if idx < 0 or idx >= metadata.total_frames:
                        self.logger.warning(f"Requested out-of-bounds frame index {idx} ignored.")
                        continue
                    
                    # Seek to index
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        raise VideoCorruptedError(f"Failed to read frame at index {idx} in {video_path}")
                    frames.append(frame)
                    
            # Mode B: Resample video temporal rates to match target FPS
            elif target_fps is not None:
                # We calculate timestamps for output frames to align audio-video sequences.
                # Example: for target_fps = 25.0, frames occur at 0.0s, 0.04s, 0.08s, etc.
                duration_sec = metadata.duration
                if self.config.video.max_duration is not None:
                    duration_sec = min(duration_sec, self.config.video.max_duration)
                num_output_frames = int(duration_sec * target_fps)
                
                for step in range(num_output_frames):
                    target_time_sec = step / target_fps
                    # Convert target time back to source frame index
                    src_frame_idx = int(round(target_time_sec * metadata.fps))
                    
                    # Prevent going out of bounds
                    if src_frame_idx >= metadata.total_frames:
                        src_frame_idx = metadata.total_frames - 1
                        
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_idx)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # Fallback: attempt to reuse the previous frame or log corruption
                        if len(frames) > 0:
                            frames.append(frames[-1])
                        else:
                            raise VideoCorruptedError(f"Failed to read initial frame at {src_frame_idx} in {video_path}")
                    else:
                        frames.append(frame)
                        
            # Mode C: Extract every single frame
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame is not None:
                        frames.append(frame)
                        
            if len(frames) == 0:
                raise VideoCorruptedError(f"Zero valid frames extracted from video: {video_path}")
                
            return frames, metadata
            
        finally:
            cap.release()

    def frame_generator(
        self,
        video_path: Path
    ) -> Generator[np.ndarray, None, None]:
        """
        Generator yielding frames one-by-one to reduce RAM usage for large video files.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoCorruptedError(f"Could not open video file for streaming: {video_path}")
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is not None:
                    yield frame
        finally:
            cap.release()


if __name__ == "__main__":
    print("Executing self-test for video_loader.py...")
    import tempfile
    
    # 1. Create a synthetic video to verify metadata loading and extraction features offline
    temp_dir = Path(tempfile.mkdtemp())
    temp_video_path = temp_dir / "synthetic_test_video.mp4"
    
    # Parameters for synthetic video
    width, height = 320, 240
    fps = 20.0
    duration = 2.0
    num_frames = int(fps * duration)
    
    # Setup OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    
    try:
        print("Generating 2-second synthetic MP4 video...")
        for i in range(num_frames):
            # Create a blank image with frame count text
            img = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                img, f"Frame {i}", (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            out.write(img)
        out.release()
        print(f"Synthetic video written to {temp_video_path}")
        
        # 2. Test VideoLoader loading and processing on the synthetic file
        loader = VideoLoader()
        
        # Test Metadata Extraction
        metadata = loader.load_metadata(temp_video_path)
        print(f"Parsed Metadata: FPS={metadata.fps}, Size={metadata.width}x{metadata.height}, "
              f"Frames={metadata.total_frames}, Duration={metadata.duration}s")
        
        assert metadata.fps == fps
        assert metadata.width == width
        assert metadata.height == height
        assert metadata.total_frames == num_frames
        
        # Test Full Extraction
        frames, meta = loader.extract_frames(temp_video_path)
        print(f"Extracted {len(frames)} frames. Target: {num_frames}")
        assert len(frames) == num_frames
        
        # Test Frame Resampling (e.g. resample 20 FPS video to 25 FPS config target)
        resampled_frames, _ = loader.extract_frames(temp_video_path, target_fps=25.0)
        expected_resampled_count = int(duration * 25.0)
        print(f"Resampled frames to 25.0 FPS: Extracted {len(resampled_frames)}. Expected: {expected_resampled_count}")
        assert len(resampled_frames) == expected_resampled_count
        
        # Test Index Specific Extraction
        idx_frames, _ = loader.extract_frames(temp_video_path, frame_indices=[0, 10, 20, 30])
        print(f"Extracted specified index frames (0, 10, 20, 30): Extracted {len(idx_frames)}")
        assert len(idx_frames) == 4
        
        print("All video_loader.py self-tests: PASSED")
        
    except Exception as e:
        print(f"Self-test failed with error: {e}")
        sys.exit(1)
        
    finally:
        # Clean up temp assets
        if temp_video_path.exists():
            temp_video_path.unlink()
        try:
            temp_dir.rmdir()
        except OSError:
            pass
