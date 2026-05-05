import os
import subprocess
import shutil

# Try to auto-locate ffmpeg – checks PATH first, then common Windows install locations
def _find_ffmpeg() -> str:
    """
    Returns the path to the ffmpeg executable.
    Raises RuntimeError with install instructions if not found.
    """
    # 1. Check if ffmpeg is on PATH
    in_path = shutil.which("ffmpeg")
    if in_path:
        return in_path

    # 2. Common manual install locations on Windows
    candidates = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "ffmpeg", "bin", "ffmpeg.exe"),
        os.path.join(os.environ.get("USERPROFILE",  ""), "ffmpeg", "bin", "ffmpeg.exe"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    raise RuntimeError(
        "FFmpeg not found on this system.\n"
        "Install it and add to PATH:\n"
        "  Windows: winget install ffmpeg\n"
        "  OR download from https://www.gyan.dev/ffmpeg/builds/ → ffmpeg-release-essentials.zip\n"
        "  Extract to C:\\ffmpeg\\  and add C:\\ffmpeg\\bin to your system PATH."
    )


FFMPEG_BIN = None  # lazy-loaded once


def extract_audio(video_path: str) -> str:
    """
    Extract audio from a video file using FFmpeg.
    Uses subprocess.run with an argument list (no shell=True) to prevent
    command injection. Returns the .wav path or raises RuntimeError on failure.
    """
    global FFMPEG_BIN
    if FFMPEG_BIN is None:
        FFMPEG_BIN = _find_ffmpeg()

    audio_path = video_path.replace(".mp4", ".wav")

    result = subprocess.run(
        [
            FFMPEG_BIN, "-y",
            "-i",    video_path,
            "-q:a",  "0",
            "-map",  "a",
            audio_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed to extract audio from '{video_path}'. "
            "Ensure the video file contains an audio track."
        )

    return audio_path