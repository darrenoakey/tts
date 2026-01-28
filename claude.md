# TTS Project

Text-to-speech using Qwen3-TTS models via mlx-audio for Apple Silicon.

## Quick Start

```bash
./run install    # Set up venv and deps
./run tts input.txt -o output.mp3
```

## Key Files

- `src/tts_engine.py` - Engine abstraction with QwenTtsEngine, VoiceDesignEngine, VoiceCloneEngine
- `src/voice_scraper.py` - Scrape voice samples from moviesoundclips.net
- `src/audio_quality.py` - Audio quality analysis, noise reduction, reference preparation
- `src/tts.py` - CLI entry point
- `voices.json` - Custom voice registry
- `run` - Project runner with venv management

## Architecture

- **MLX backend**: Uses mlx-audio for efficient Apple Silicon inference
- **Chunked processing**: Long texts split into ~3 sentence chunks to prevent memory spikes
- **File locking**: Only one TTS instance runs at a time (uses `.tts.lock`)
- **Lazy model loading**: Model loaded on first synthesis, reused across calls

## Voices

**Built-in voices** (CustomVoice model):
- English: aiden, ryan, ono_anna, sohee
- Chinese: vivian, serena, uncle_fu, dylan (beijing dialect), eric (sichuan dialect)

**Custom voices** stored in `voices.json`:
- Description-based: Uses VoiceDesign model with text description of desired voice
- Clone-based: Uses Base model with ref_audio + ref_text for voice cloning

**Voice cloning requirements**:
- `ref_audio`: Path to reference audio file (24kHz sample rate)
- `ref_text`: Transcription of what's said in reference audio (required - without it, generation hangs)
- Optimal reference length: 10-15 seconds of clean speech
- Use mlx-whisper to transcribe: `mlx_whisper.transcribe(audio_path, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")`

**Audio quality for voice cloning**:
- Use `src/audio_quality.py` to analyze clips by SNR and select best quality
- Noise reduction applied automatically with noisereduce library
- Run `python -m src.audio_quality <clip_dir> <max_duration>` to prepare clean reference

## Models

Default: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`

Smaller/faster: `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16`

## Gotchas

- Model downloads ~5GB on first run (cached in `~/.cache/huggingface/`)
- Memory monitoring script (`run_with_memory_monitor.sh`) useful for debugging OOM issues
- **Sample rate is 24kHz** (not 12kHz - the "12Hz" in model name is token rate)
- Audio normalization uses ffmpeg `loudnorm` and `alimiter` filters to prevent clipping
- Temperature parameter (`-t`) controls synthesis variability (default 0.9)
