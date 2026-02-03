"""
============================================================
 AdGuard – AI-Based Video Compliance System
 PHASE 1 | Module: video_preprocessor
============================================================
 Purpose   : Ingest a raw video file and produce sampled
             frames (with timestamps), a clean mono WAV
             audio track, and a metadata dictionary.

 Outputs   :
   • frames/          – directory of sampled frame images
   • audio.wav        – mono, 16 kHz audio track
   • metadata.json    – duration, original FPS, sample FPS,
                        frame-to-timestamp map
============================================================
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List

import cv2                          # OpenCV  – frame extraction
import subprocess                   # FFmpeg  – audio extraction (subprocess wrapper)
import numpy as np                  # NumPy   – lightweight array ops (optional usage)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("video_preprocessor")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FrameInfo:
    """Holds metadata for a single sampled frame."""
    frame_id: int           # sequential index (0-based)
    timestamp_sec: float    # time in seconds (rounded to 3 dp)
    file_path: str          # absolute path to saved .png


@dataclass
class PreprocessorOutput:
    """Top-level output handed to the next pipeline stage."""
    video_path: str
    duration_sec: float
    original_fps: float
    sample_fps: float                       # FPS chosen for extraction
    frames: List[FrameInfo] = field(default_factory=list)
    audio_path: str = ""                    # path to extracted .wav
    metadata_path: str = ""                 # path to metadata.json


# ---------------------------------------------------------------------------
# Configuration – tweak these constants as needed
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_FPS: float = 2.0            # extract 2 frames / second
AUDIO_SAMPLE_RATE: int   = 16000           # 16 kHz – good for ASR engines
AUDIO_CHANNELS: int      = 1               # mono
SUPPORTED_EXTENSIONS     = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


# ===========================================================================
# Core functions
# ===========================================================================

def validate_video_path(video_path: str) -> Path:
    """
    Ensure the supplied path points to an existing, supported video file.

    Args:
        video_path: String path to the video.

    Returns:
        Resolved pathlib.Path object.

    Raises:
        FileNotFoundError : file does not exist.
        ValueError        : extension not in SUPPORTED_EXTENSIONS.
    """
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{path.suffix}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )
    logger.info("Video validated: %s", path)
    return path


def get_video_metadata(video_path: Path) -> dict:
    """
    Read FPS and total duration from the video using OpenCV.

    Args:
        video_path: Validated path to the video.

    Returns:
        dict with keys:
            fps      (float) – frames per second of the source video
            duration (float) – total length in seconds
            frame_count (int)

    Raises:
        IOError: if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration    = frame_count / fps if fps > 0 else 0.0

    cap.release()

    metadata = {
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration": round(duration, 3),
    }
    logger.info("Metadata extracted: %s", metadata)
    return metadata


def sample_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    original_fps: float = 30.0,
) -> List[FrameInfo]:
    """
    Walk through the video and grab one frame every (1 / sample_fps) seconds.
    Each frame is saved as a PNG and mapped to its timestamp.

    Args:
        video_path  : Path to the video file.
        output_dir  : Directory where frame PNGs will be saved.
        sample_fps  : How many frames to capture per second (1–5 recommended).
        original_fps: Native FPS of the source video (used to compute step).

    Returns:
        List of FrameInfo dataclasses (one per saved frame).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open video: {video_path}")

    # Number of native frames to skip between each sampled frame
    frame_step = max(1, int(original_fps / sample_fps))

    frames_info: List[FrameInfo] = []
    current_frame_idx = 0          # native frame counter
    sampled_idx       = 0          # our sequential output index

    while True:
        # Jump to the target native frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()

        if not ret:
            break                  # end of video

        # Calculate the real-world timestamp for this frame
        timestamp = round(current_frame_idx / original_fps, 3)

        # Save frame to disk
        frame_filename = f"frame_{sampled_idx:05d}.png"
        frame_path     = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

        frames_info.append(
            FrameInfo(
                frame_id=sampled_idx,
                timestamp_sec=timestamp,
                file_path=str(frame_path),
            )
        )

        logger.debug("Saved %s at %.3f s", frame_filename, timestamp)

        # Advance by frame_step
        current_frame_idx += frame_step
        sampled_idx       += 1

    cap.release()
    logger.info("Sampled %d frames at %.1f FPS.", len(frames_info), sample_fps)
    return frames_info


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int    = AUDIO_CHANNELS,
) -> str:
    """
    Use FFmpeg (CLI) to pull the audio stream out of the video and save it
    as a clean mono WAV — ready for ASR in Phase 2.

    Args:
        video_path  : Source video.
        output_path : Where to write the .wav file.
        sample_rate : Target sample rate in Hz.
        channels    : 1 = mono, 2 = stereo.

    Returns:
        String path to the generated WAV file.

    Raises:
        FileNotFoundError : FFmpeg is not installed / not on PATH.
        subprocess.CalledProcessError : FFmpeg returned a non-zero exit code.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                           # overwrite output without asking
        "-i", str(video_path),          # input
        "-vn",                          # no video stream in output
        "-acodec", "pcm_s16le",         # 16-bit PCM (standard WAV)
        "-ar", str(sample_rate),        # sample rate
        "-ac", str(channels),           # mono
        str(output_path),               # output
    ]

    logger.info("Extracting audio with FFmpeg …")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            "FFmpeg is not installed or not on your system PATH. "
            "Install it from https://ffmpeg.org/download.html"
        )

    logger.info("Audio saved to %s", output_path)
    return str(output_path)


def save_metadata(output: PreprocessorOutput, output_dir: Path) -> str:
    """
    Persist the full PreprocessorOutput as a JSON file so downstream modules
    can reload it without re-running extraction.

    Args:
        output     : Completed PreprocessorOutput instance.
        output_dir : Directory for the JSON file.

    Returns:
        Path string to the saved metadata.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.json"

    # Convert dataclass tree → plain dict
    payload = asdict(output)

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Metadata saved to %s", meta_path)
    return str(meta_path)


# ===========================================================================
# Main orchestrator
# ===========================================================================

def preprocess_video(
    video_path: str,
    output_base_dir: str = "preprocessed_output",
    sample_fps: float    = DEFAULT_SAMPLE_FPS,
) -> PreprocessorOutput:
    """
    Single entry-point that ties every step together.

    Pipeline:
        1. Validate input path
        2. Read native FPS & duration
        3. Sample frames → frames/
        4. Extract audio  → audio/audio.wav
        5. Write metadata → metadata.json
        6. Return PreprocessorOutput

    Args:
        video_path      : Path to the raw video file.
        output_base_dir : Root folder for all generated assets.
        sample_fps      : Frames per second to extract (1–5).

    Returns:
        Fully populated PreprocessorOutput dataclass.

    Example:
        >>> result = preprocess_video("ad_clip.mp4", sample_fps=2)
        >>> print(f"Extracted {len(result.frames)} frames")
        >>> print(f"Audio at  {result.audio_path}")
    """
    # --- 1. Validate ---------------------------------------------------
    video_path_obj = validate_video_path(video_path)

    # --- 2. Metadata ---------------------------------------------------
    meta        = get_video_metadata(video_path_obj)
    original_fps = meta["fps"]
    duration     = meta["duration"]

    base = Path(output_base_dir).resolve()

    # --- 3. Sample frames ----------------------------------------------
    frames_dir  = base / "frames"
    frames_info = sample_frames(
        video_path=video_path_obj,
        output_dir=frames_dir,
        sample_fps=sample_fps,
        original_fps=original_fps,
    )

    # --- 4. Extract audio ----------------------------------------------
    audio_path = extract_audio(
        video_path=video_path_obj,
        output_path=base / "audio" / "audio.wav",
    )

    # --- 5. Assemble output --------------------------------------------
    result = PreprocessorOutput(
        video_path   = str(video_path_obj),
        duration_sec = duration,
        original_fps = original_fps,
        sample_fps   = sample_fps,
        frames       = frames_info,
        audio_path   = audio_path,
    )

    # --- 6. Persist metadata -------------------------------------------
    result.metadata_path = save_metadata(result, base)

    logger.info(
        "Preprocessing complete | duration=%.2fs | frames=%d | audio=%s",
        result.duration_sec,
        len(result.frames),
        result.audio_path,
    )
    return result


# ===========================================================================
# CLI runner (quick smoke-test)
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AdGuard – Video Preprocessor (Phase 1)"
    )
    parser.add_argument("video",      help="Path to the input video file.")
    parser.add_argument(
        "-o", "--output",
        default="preprocessed_output",
        help="Output directory (default: preprocessed_output/)",
    )
    parser.add_argument(
        "-f", "--fps",
        type=float,
        default=DEFAULT_SAMPLE_FPS,
        help="Sampling rate in frames/sec (default: 2.0)",
    )

    args = parser.parse_args()

    output = preprocess_video(
        video_path      = args.video,
        output_base_dir = args.output,
        sample_fps      = args.fps,
    )

    print("\n✅  Done!")
    print(f"   Frames   → {Path(output.frames[0].file_path).parent}")
    print(f"   Audio    → {output.audio_path}")
    print(f"   Metadata → {output.metadata_path}")
