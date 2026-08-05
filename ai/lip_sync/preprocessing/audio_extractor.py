"""
audio_extractor.py

This module contains the AudioExtractor and supporting functions for extracting,
converting, and normalizing audio tracks from video files. It leverages FFmpeg to
demux and transcode audio to 16 kHz Mono PCM, and uses soundfile/numpy to perform
peak amplitude normalization.

If FFmpeg is not found in the system PATH, the class automatically downloads a static
Windows build of `ffmpeg.exe` from Hugging Face to the checkpoints directory to ensure
offline execution support after the first run.
"""

import logging
from pathlib import Path
import subprocess
import sys
import urllib.request
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Ensure project root is in path for direct execution
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai.lip_sync.preprocessing.config import ProjectConfig, get_default_config


class AudioExtractionError(Exception):
    """Raised when audio extraction or conversion processes fail."""
    pass


class AudioExtractor:
    """
    Handles audio extraction from video files using FFmpeg and performs amplitude normalization.
    Automatically downloads a static FFmpeg binary for Windows if not locally available.
    """

    FFMPEG_DOWNLOAD_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/ffmpeg.exe"

    def __init__(self, config: Optional[ProjectConfig] = None):
        """
        Initializes the AudioExtractor with config parameters.
        """
        self.config = config or get_default_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ffmpeg_cmd = "ffmpeg"  # Default fallback command
        
        # Verify and set up ffmpeg path on init
        self._resolve_ffmpeg()

    def _resolve_ffmpeg(self) -> None:
        """
        Resolves the FFmpeg executable command. First tries system-wide 'ffmpeg',
        then falls back to downloading a static binary to the checkpoints directory.
        """
        # 1. Try system-wide 'ffmpeg'
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=3.0
            )
            self.ffmpeg_cmd = "ffmpeg"
            self.logger.info("Found system-wide FFmpeg installation in PATH.")
            return
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.info("System-wide FFmpeg not found in PATH. Checking checkpoints directory...")

        # 2. Check checkpoints folder for downloaded ffmpeg.exe
        local_ffmpeg = self.config.paths.checkpoints_dir / "ffmpeg.exe"
        if local_ffmpeg.exists():
            try:
                subprocess.run(
                    [str(local_ffmpeg), "-version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    timeout=3.0
                )
                self.ffmpeg_cmd = str(local_ffmpeg)
                self.logger.info(f"Using local static FFmpeg binary at: {local_ffmpeg}")
                return
            except subprocess.SubprocessError as e:
                self.logger.warning(f"Local FFmpeg binary found but not functioning: {e}. Re-downloading...")
                local_ffmpeg.unlink(missing_ok=True)

        # 3. Download static build if offline is not requested/ready
        self.logger.warning("FFmpeg executable not found. Starting automatic download from Hugging Face...")
        self.config.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self._download_ffmpeg(local_ffmpeg)
            
            # Verify the downloaded binary
            subprocess.run(
                [str(local_ffmpeg), "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=3.0
            )
            self.ffmpeg_cmd = str(local_ffmpeg)
            self.logger.info(f"FFmpeg downloaded and verified successfully. Path: {local_ffmpeg}")
        except Exception as e:
            raise AudioExtractionError(
                f"Failed to automatically obtain and verify FFmpeg. "
                f"Please install FFmpeg manually or add it to PATH. Details: {e}"
            )

    def _download_ffmpeg(self, dest_path: Path) -> None:
        """
        Downloads static ffmpeg.exe from Hugging Face with progress tracking.
        """
        self.logger.info(f"Downloading static FFmpeg from: {self.FFMPEG_DOWNLOAD_URL}")
        
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        # Request user approval might be needed or runs natively since we have read_url(*)
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="ffmpeg.exe") as t:
                urllib.request.urlretrieve(
                    self.FFMPEG_DOWNLOAD_URL,
                    filename=str(dest_path),
                    reporthook=t.update_to
                )
        except Exception as e:
            # Clean up partial download
            if dest_path.exists():
                dest_path.unlink()
            raise e

    def extract_audio(self, video_path: Path, output_wav_path: Path) -> Path:
        """
        Extracts the audio track from the video file using FFmpeg, resampling
        it to 16 kHz Mono PCM WAV format. Performs subsequent normalization.
        
        Args:
            video_path (Path): Path to the source video file.
            output_wav_path (Path): Destination path for the output WAV file.
            
        Returns:
            Path: Absolute path to the extracted and normalized WAV file.
            
        Raises:
            FileNotFoundError: If the source video file does not exist.
            AudioExtractionError: If the FFmpeg subprocess fails.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found for audio extraction: {video_path}")
            
        # Ensure parent directory of output WAV exists
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg command arguments:
        # -y : Overwrite output file
        # -i : Input video file
        # -vn : Disable video stream decoding
        # -acodec pcm_s16le : Output format 16-bit signed PCM
        # -ar : Resample to target sample rate (e.g. 16000 Hz)
        # -ac : Set audio channels to target (e.g. 1 mono)
        cmd = [
            self.ffmpeg_cmd,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.config.audio.sample_rate),
            "-ac", str(self.config.audio.channels),
            str(output_wav_path)
        ]
        
        self.logger.info(f"Extracting audio from {video_path.name} to {output_wav_path.name}...")
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            self.logger.debug(f"FFmpeg stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg extraction failed. Stderr: {e.stderr}")
            raise AudioExtractionError(f"FFmpeg failed with exit code {e.returncode}. Stderr: {e.stderr}")
            
        # Perform peak normalization using soundfile
        if self.config.audio.normalize:
            self.normalize_wav(output_wav_path)
            
        return output_wav_path

    def normalize_wav(self, wav_path: Path) -> None:
        """
        Performs peak amplitude normalization on a WAV file.
        Scales the audio signals so the absolute peak is 0.99 (leaving headroom).
        """
        try:
            data, sample_rate = sf.read(str(wav_path))
            
            # If stereo/multi-channel somehow occurred, average channels to mono
            if len(data.shape) > 1:
                self.logger.warning(f"Audio file {wav_path.name} has multiple channels. Downmixing to mono.")
                data = np.mean(data, axis=1)
                
            max_val = np.max(np.abs(data))
            if max_val > 0:
                # Standard peak normalization
                normalized_data = (data / max_val) * 0.99
                sf.write(str(wav_path), normalized_data, sample_rate)
                self.logger.debug(f"Peak normalized {wav_path.name} from max absolute amplitude {max_val:.4f} to 0.99.")
            else:
                self.logger.warning(f"Audio file {wav_path.name} contains only silence. Normalization skipped.")
        except Exception as e:
            self.logger.error(f"Failed to normalize WAV file: {wav_path}. Error: {e}")
            raise AudioExtractionError(f"Failed to peak normalize audio: {e}")

    def load_waveform(self, wav_path: Path) -> Tuple[np.ndarray, int]:
        """
        Utility to load WAV file content as a float32 array and return its sample rate.
        """
        try:
            data, sample_rate = sf.read(str(wav_path), dtype="float32")
            return data, sample_rate
        except Exception as e:
            self.logger.error(f"Failed to read WAV file: {wav_path}. Error: {e}")
            raise AudioExtractionError(f"Could not load waveform from WAV: {e}")


if __name__ == "__main__":
    print("Executing self-test for audio_extractor.py...")
    import tempfile
    
    # Initialize extractor (this will trigger download if not found)
    try:
        extractor = AudioExtractor()
    except AudioExtractionError as e:
        print(f"Failed to initialize AudioExtractor: {e}", file=sys.stderr)
        sys.exit(1)
        
    temp_dir = Path(tempfile.mkdtemp())
    temp_wav_path = temp_dir / "synthetic_audio.wav"
    temp_video_path = temp_dir / "synthetic_video.mp4"
    extracted_wav_path = temp_dir / "extracted_audio.wav"
    
    try:
        # 1. Generate a synthetic audio sine wave
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Low amplitude sine wave (max absolute value 0.5) to test normalization
        audio_signal = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        
        # Save as WAV
        sf.write(str(temp_wav_path), audio_signal, sample_rate)
        print(f"Generated synthetic WAV file at: {temp_wav_path}")
        
        # Verify normalization utility works on its own
        extractor.normalize_wav(temp_wav_path)
        norm_data, norm_sr = extractor.load_waveform(temp_wav_path)
        max_amplitude = np.max(np.abs(norm_data))
        print(f"Normalized WAV max amplitude: {max_amplitude:.4f}. Expected: ~0.99")
        assert np.allclose(max_amplitude, 0.99, atol=1e-3)
        assert norm_sr == sample_rate
        
        # 2. Create a synthetic video container with audio to test extraction
        # We use FFmpeg to generate a dummy video containing the audio
        print("Generating a test video with audio using FFmpeg...")
        cmd = [
            extractor.ffmpeg_cmd,
            "-y",
            "-f", "lavfi",
            "-i", "color=c=black:s=320x240:d=2.0",
            "-i", str(temp_wav_path),
            "-c:v", "mpeg4",
            "-c:a", "aac",
            "-shortest",
            str(temp_video_path)
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Test video created at: {temp_video_path}")
        
        # 3. Extract audio back from the video
        extractor.extract_audio(temp_video_path, extracted_wav_path)
        print(f"Extracted audio saved to: {extracted_wav_path}")
        
        # Verify extracted properties
        ext_data, ext_sr = extractor.load_waveform(extracted_wav_path)
        print(f"Extracted properties: samples={len(ext_data)}, sample_rate={ext_sr}, mono={len(ext_data.shape)==1}")
        assert ext_sr == 16000
        assert len(ext_data.shape) == 1
        assert np.max(np.abs(ext_data)) > 0.90  # check if normalized
        
        print("All audio_extractor.py self-tests: PASSED")
        
    except Exception as e:
        print(f"Self-test failed with error: {e}", file=sys.stderr)
        sys.exit(1)
        
    finally:
        # Clean up temp assets
        for path in [temp_wav_path, temp_video_path, extracted_wav_path]:
            if path.exists():
                path.unlink()
        try:
            temp_dir.rmdir()
        except OSError:
            pass
