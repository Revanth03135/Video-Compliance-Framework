from faster_whisper import WhisperModel


def transcribe_audio(audio_path, model_size="base", word_level=False):
    """
    Transcribes an audio file and returns timestamped transcript.

    Parameters:
        audio_path (str): Path to audio file
        model_size (str): tiny, base, small, medium, large-v3
        word_level (bool): If True, returns word-level timestamps

    Returns:
        str: Timestamped transcript
    """
    import os

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Select device safely
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    # Choose compute type based on device
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )
    # Whisper supports most audio formats via ffmpeg (mp3, wav, m4a, flac, ogg, mp4, etc.)
    # Make sure ffmpeg is installed on your system.
    try:
        segments, _ = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=word_level
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to process audio file. Ensure ffmpeg is installed "
            "and the audio format is supported."
        ) from e

    def format_time(seconds):
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hrs:02}:{mins:02}:{secs:02}"

    transcript_lines = []

    if word_level:
        for segment in segments:
            for word in segment.words:
                start = format_time(word.start)
                end = format_time(word.end)
                transcript_lines.append(
                    f"[{start} - {end}] {word.word}"
                )
    else:
        for segment in segments:
            start = format_time(segment.start)
            end = format_time(segment.end)
            text = segment.text.strip()
            transcript_lines.append(
                f"[{start} - {end}] {text}"
            )

    return "\n".join(transcript_lines)
