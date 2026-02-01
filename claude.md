# TTS Project

Text-to-speech using Qwen3-TTS models via mlx-audio for Apple Silicon.

## Quick Start

```bash
./run install    # Set up venv and deps
./run tts input.txt -o output.mp3
./run tts "Hello world" -v aiden   # Inline text (no file needed)
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
- **Chunked processing**: Long texts split into ~600 word chunks (at sentence boundaries) to prevent memory spikes and avoid 1.7B model degradation
- **Subprocess isolation**: Model runs in isolated subprocess, exits cleanly when done
  - QwenTtsEngine/VoiceCloneEngine: ONE subprocess for ALL chunks (avoids asyncio SIGCHLD conflicts with autoblog)
  - VoiceDesignEngine: one subprocess per chunk (supports resumability via work_dir)
- **Memory monitoring**: Subprocess tracks RSS via `resource.getrusage()`. If exceeds 10GB, exits with code 77 and parent restarts from next chunk
- **Memory safety checks**: Before starting, available system memory is checked. If below 1.5GB threshold, synthesis aborts with clear error
- **File locking**: Only one TTS instance runs at a time (uses `.tts.lock`)

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
- **Reference audio should be ~2 minutes** - longer causes memory issues and slowdowns
- Use mlx-whisper to transcribe: `mlx_whisper.transcribe(audio_path, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")`

**Audio quality for voice cloning**:
- Use `src/audio_quality.py` to analyze clips by SNR and select best quality
- AI enhancement via resemble-enhance with quality modes:
  - `--fast`: nfe=64 (fastest, lowest quality)
  - `--hq`: nfe=96
  - Default: `--ultra` nfe=128 (max allowed by resemble-enhance)
- Run `python -m src.audio_quality <clip_dir> <max_duration> [quality]` to prepare clean reference

**Voice commands**:
- `./run design-voice myvoice "description..."` - Create voice from text description (generates template, saves as clone)
- `./run clone-voice myvoice recording.wav` - Create cloned voice from your own audio
- `./run scrape-voice "Nathan Fillion"` - Create cloned voice from celebrity
- `./run list-celebs` - List all 58 available celebrities on moviesoundclips.net
- `./run list-voices` - Show built-in, designed, and cloned voices

**Voice training scripts**:
- `script_30.txt` - 30-second script (~45 words) - quick voice tests
- `script.txt` - 2-minute phonetically balanced text (~270 words) - use for voice design
- `script16.txt` - 16-minute extended script (~720 words) - for maximum voice capture

## Models

**CustomVoice** (preset voices): `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`

**VoiceDesign** (designed voices): `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`

**Voice Cloning** (Base models):
- 0.6B: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16` (faster)
- 1.7B: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` (same speed up to ~500 words, degrades beyond that, **default**)
- All models use bf16 (bfloat16) - full precision, NOT quantized

**Model performance** (benchmarked with gary voice):
- Both models: ~1.1-1.3 seconds per word up to 512 words
- 1.7B degrades to 2.24 s/word at 1024 words (1.76x slower than 0.6B)
- Chunking at 600 words keeps both models in optimal range

## Gotchas

- Model downloads ~5GB on first run (cached in `~/.cache/huggingface/`)
- Memory monitoring script (`run_with_memory_monitor.sh`) useful for debugging OOM issues
- **Sample rate is 24kHz** (not 12kHz - the "12Hz" in model name is token rate)
- **MP3 output uses maximum quality** (qscale:a 0, ~245 kbps VBR)
- Audio normalization uses ffmpeg `loudnorm` and `alimiter` filters to prevent clipping
- Temperature parameter (`-t`) controls synthesis variability (default 0.9)
- **SSML not supported** - Qwen3-TTS uses natural language instructions for prosody, not XML tags
- Reference audio is auto-resampled from any format to 24kHz mono (cached after first use)
- Stereo recordings are automatically converted to mono for voice cloning
- Voice scraping uses **ultra quality enhancement by default** (nfe=128, slowest but best)
- **VoiceDesign 1.7B is very slow** - expect ~1 hour per 3-sentence chunk with enhancement
- **Chunk-level enhancement** - enhancement runs per-chunk (faster than enhancing combined file)
- **Temp file cleanup** - if TTS is killed mid-process, orphan wav files remain in /var/folders; clean manually if needed
- **Enhancement output is 44.1kHz** - resemble-enhance upsamples from 24kHz to 44.1kHz
- **NEVER run multiple models in parallel** - causes GPU crashes (Metal command buffer errors). Always run one model at a time, sequentially
- **design-voice is resumable** - if it crashes, just run again and it will skip completed chunks
- **design-voice auto-retries** - on GPU crash, will automatically retry up to 10 times
- **Memory auto-restart** - if subprocess exceeds 10GB RSS, it exits with code 77 and parent restarts from next chunk (up to 50 restarts)
- **asyncio compatibility** - single subprocess for all chunks avoids SIGCHLD signal conflicts when called from asyncio apps (like autoblog)
