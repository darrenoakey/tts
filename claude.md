# TTS Project

Text-to-speech using Qwen3-TTS models via mlx-audio for Apple Silicon.

## Quick Start

```bash
./run install    # Set up venv and deps
./run tts input.txt -o output.mp3
```

## Key Files

- `src/tts_engine.py` - Engine abstraction and QwenTtsEngine implementation
- `src/tts.py` - CLI entry point
- `run` - Project runner with venv management

## Architecture

- **MLX backend**: Uses mlx-audio for efficient Apple Silicon inference
- **Chunked processing**: Long texts split into ~3 sentence chunks to prevent memory spikes
- **File locking**: Only one TTS instance runs at a time (uses `.tts.lock`)
- **Lazy model loading**: Model loaded on first synthesis, reused across calls

## Voices

Available for CustomVoice model:
- English: aiden, ryan
- Chinese: vivian, serena, uncle_fu, dylan, eric

## Models

Default: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`

Smaller/faster: `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16`

## Gotchas

- Model downloads ~5GB on first run (cached in `~/.cache/huggingface/`)
- Memory monitoring script (`run_with_memory_monitor.sh`) useful for debugging OOM issues
