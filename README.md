# Video Preprocessor – Setup & Run Guide

This is **Phase 1** of the AdGuard AI Video Compliance System. It takes a raw video file and extracts sampled frames, a clean audio track, and metadata for the next stages of the pipeline.

---

## Dependencies to Install

### 1. Python (3.8 or higher)
Make sure Python is installed on your machine. You can check by running:
```bash
python --version
```

---

### 2. FFmpeg (system-level install)
FFmpeg is **not** installed via pip. You need to install it separately on your machine.

**Windows:**
- Download from https://ffmpeg.org/download.html
- Extract the folder anywhere (e.g. `C:/ffmpeg`)
- Add `C:/ffmpeg/bin` to your **System PATH**
- Verify by running:
```bash
ffmpeg -version
```

**Mac (using Homebrew):**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

---

### 3. Python Packages
Run this single command in your terminal to install everything:
```bash
pip install opencv-python numpy
```

---

## Supported Video Formats
The script accepts the following formats:
- `.mp4`
- `.mov`
- `.mkv`
- `.avi`
- `.webm`

---

## How to Run

### Option A – Command Line (Recommended)

Basic usage:
```bash
python video_preprocessor.py <path_to_your_video>
```

Example:
```bash
python video_preprocessor.py ad_clip.mp4
```

With custom output folder and frame rate:
```bash
python video_preprocessor.py ad_clip.mp4 -o my_output -f 3
```

| Flag | What it does | Default |
|------|-------------|---------|
| `-o` | Sets the output folder name | `preprocessed_output` |
| `-f` | Sets how many frames to extract per second (1–5) | `2` |

---

### Option B – Inside a Python Script

```python
from video_preprocessor import preprocess_video

result = preprocess_video(
    video_path="ad_clip.mp4",
    output_base_dir="my_output",
    sample_fps=2
)

print(result.frames)       # all extracted frames + timestamps
print(result.audio_path)   # path to the extracted audio
print(result.metadata_path) # path to metadata.json
```

---

## What Gets Generated

After a successful run, your output folder will look like this:

```
preprocessed_output/
├── frames/
│   ├── frame_00000.png
│   ├── frame_00001.png
│   ├── frame_00002.png
│   └── ...
├── audio/
│   └── audio.wav
└── metadata.json
```

| Output | Description |
|--------|-------------|
| `frames/` | Extracted frame images, each mapped to a timestamp |
| `audio/audio.wav` | Clean mono audio at 16 kHz – ready for speech recognition |
| `metadata.json` | Full details: duration, FPS, frame timestamps, file paths |

---

## Quick Checklist Before Running

- [ ] Python 3.8+ is installed
- [ ] FFmpeg is installed and visible in your PATH (run `ffmpeg -version` to confirm)
- [ ] Python packages are installed (`pip install opencv-python numpy`)
- [ ] Your video file is in a supported format (.mp4, .mov, .mkv, .avi, .webm)

---

## Troubleshooting

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `FFmpeg is not installed or not on your system PATH` | FFmpeg not added to PATH | Re-check your PATH setting or reinstall FFmpeg |
| `Video file not found` | Wrong file path | Double-check the path you passed in |
| `Unsupported video format` | File extension not supported | Convert your video to `.mp4` using FFmpeg |
| `OpenCV cannot open video` | Corrupted file or missing codec | Try re-encoding the video with FFmpeg first |
