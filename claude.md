# TTS Project

Text-to-speech using Qwen3-TTS models.

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

- **Chunked processing**: Long texts split into ~3 sentence chunks to prevent memory spikes
- **File locking**: Only one TTS instance runs at a time (uses `.tts.lock`)
- **Lazy model loading**: Model loaded on first synthesis, reused across calls

## Voices

Available for CustomVoice model: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian

## Gotchas

- MPS (Apple Silicon) falls back to CPU - MPS support for qwen-tts is experimental
- Flash attention only used on CUDA, otherwise uses standard attention via shim
- Memory monitoring script (`run_with_memory_monitor.sh`) useful for debugging OOM issues
