# Authentix - Lip Sync Detection Module

This module contains a production-grade Lip Sync Preprocessing and Model Integration pipeline for the Authentix multimodal deepfake detection platform. It handles resampled video decoding, peak-normalized audio extraction, active speaker face tracking, facial mesh landmarks, mouth ROI crop sequences, and MFCC features alignment into sliding-window PyTorch tensors ready for `lithiumice/syncnet` model evaluation.

## Project Structure

```text
ai/
└── lip_sync/
    ├── dataset/
    │   └── processed/          # Contains outputs (crops, WAV, tensors, metadata.json)
    ├── preprocessing/
    │   ├── config.py           # Configuration definitions and initializations
    │   ├── video_loader.py     # Decodes and resamples video frames to target FPS (25 FPS)
    │   ├── audio_extractor.py   # Extracts & peak-normalizes mono audio WAV via FFmpeg
    │   ├── face_detector.py    # S3FD face detector & speaker tracking (sfd_face.pth)
    │   ├── landmark_detector.py # MediaPipe FaceMesh landmarks (face_landmarker.task)
    │   ├── lip_cropper.py      # Crops & normalizes mouth ROIs to 111x111 grayscale
    │   └── sync_preprocessor.py # End-to-end pipeline orchestrator & aligner
    ├── checkpoints/            # Downloaded model weights (sfd_face.pth, face_landmarker.task, ffmpeg.exe)
    ├── logs/                   # System execution logs (lip_sync_preprocessing.log)
    └── README.md               # Documentation
```

## Setup & Dependencies

The module runs on **Python 3.11** and relies on the following key dependencies (already present in the project's virtual environment):
- OpenCV (`opencv-python`)
- PyTorch (`torch`)
- Torchaudio (`torchaudio`)
- MediaPipe (`mediapipe` v1.0.0+)
- SoundFile (`soundfile`)
- NumPy (`numpy`)
- tqdm (`tqdm`)
- FFmpeg (Windows binary downloaded automatically)

### Pretrained Weights
The following weights are required and will be **automatically downloaded** on first run:
1. `sfd_face.pth` (from Hugging Face `lithiumice/syncnet`)
2. `face_landmarker.task` (from Google's official Storage CDN)
3. `ffmpeg.exe` (from Hugging Face static mirror, downloaded if FFmpeg is not found in the host system's PATH)

## Usage

To run the end-to-end preprocessing pipeline on a video file:

```python
from pathlib import Path
from ai.lip_sync.preprocessing.sync_preprocessor import SyncPreprocessor
from ai.lip_sync.preprocessing.config import get_default_config

# 1. Load config
config = get_default_config()

# 2. Initialize preprocessor (downloads weights and binaries if missing)
preprocessor = SyncPreprocessor(config)

# 3. Process video
video_path = Path("path/to/your/video.mp4")
report = preprocessor.process(video_path, output_subdir_name="my_sync_run")

print(f"Video input tensor saved to: {report['video_tensor_path']}")
print(f"Audio input tensor saved to: {report['audio_tensor_path']}")
print(f"Metadata manifest saved to: {report['metadata_path']}")
```

## Generated Outputs

Under `ai/lip_sync/dataset/processed/<subdir>/`, the pipeline produces:
1. **`video_input.pt`**: A PyTorch tensor of shape `(Batch, 1, 5, 111, 111)`. Contains sliding windows of 5 consecutive grayscale mouth crop frames normalized to float $[0.0, 1.0]$.
2. **`audio_input.pt`**: A PyTorch tensor of shape `(Batch, 1, 20, 13)`. Contains corresponding 20-frame slices of 13-dimensional MFCC audio features extracted at $100\text{ Hz}$.
3. **`audio.wav`**: Peak-normalized $16\text{ kHz}$ mono WAV file.
4. **`mouth_crops/`**: Folder containing individual cropped mouth frames saved as JPEG (`mouth_00000.jpg`, etc.) for visual validation.
5. **`metadata.json`**: Manifest detail showing resampled parameters, shapes, frame counts, and original file info.

## Verification & Self-Testing

To run verification self-tests on individual modules, execute:

```powershell
# Test configuration and folders creation
.\.venv\Scripts\python.exe ai/lip_sync/preprocessing/config.py

# Test video loader and temporal resampling
.\.venv\Scripts\python.exe ai/lip_sync/preprocessing/video_loader.py

# Test FFmpeg resolving, download, and audio normalization
.\.venv\Scripts\python.exe ai/lip_sync/preprocessing/audio_extractor.py

# Test S3FD face detector network and active speaker tracking
.\.venv\Scripts\python.exe ai/lip_sync/preprocessing/face_detector.py

# Test MediaPipe FaceLandmarker and lips dynamic connections mapping
.\.venv\Scripts\python.exe ai/lip_sync/preprocessing/landmark_detector.py

# Test mouth ROI cropping and grayscale resizing
.\.venv\Scripts\python.exe ai/lip_sync/preprocessing/lip_cropper.py

# Run end-to-end integration self-test
.\.venv\Scripts\python.exe ai/lip_sync/preprocessing/sync_preprocessor.py
```
