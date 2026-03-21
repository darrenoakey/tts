# TTS Project

Text-to-speech using Qwen3-TTS models via arbiter (NVIDIA CUDA on spark).

## Quick Start

```bash
./run install    # Set up venv and deps
./run tts input.txt -o output.mp3
./run tts "Hello world" -v aiden   # Inline text (no file needed)
```

## Key Files

- `src/arbiter_engine.py` - **Default engine**: delegates TTS to arbiter HTTP API on spark
- `src/tts_engine.py` - Engine abstraction, chunking, audio processing (shared utilities)
- `src/spark_engine.py` - Direct SSH/SCP engine (library code, not default)
- `src/spark_worker.py` - Worker script that runs on spark for direct SSH mode
- `src/voice_scraper.py` - Scrape voice samples from moviesoundclips.net
- `src/audio_quality.py` - Audio quality analysis, noise reduction, reference preparation
- `src/tts.py` - CLI entry point (list-voices only; tts/multi routed via run)
- `src/voices.json` - Custom voice registry
- `run` - Project runner with venv management

## Architecture

- **Arbiter backend (default)**: All TTS commands route through the arbiter HTTP API on spark (10.0.0.254:8400)
- **PyTorch + CUDA**: Arbiter uses `qwen-tts` (official PyTorch package) with `Qwen/Qwen3-TTS-*` models on NVIDIA GB10
- **Parallel chunk submission**: All chunks submitted to arbiter queue at once, then polled until complete — eliminates per-chunk round-trip latency
- **Arbiter queuing**: Arbiter serializes jobs per model; no local file locking needed
- **Chunked processing**: Long texts split into ~600 word chunks (at sentence boundaries) — model degrades beyond this even on CUDA
- **Local MLX fallback**: `src/tts_engine.py` still contains full local MLX engines if needed
- **Local post-processing**: MP3 conversion, normalization, speed adjustment, and enhancement all happen locally via ffmpeg

## Voices

**Built-in voices** (CustomVoice model):
- English: aiden, ryan, ono_anna, sohee
- Chinese: vivian, serena, uncle_fu, dylan (beijing dialect), eric (sichuan dialect)

**Custom voices** stored in `src/voices.json`:
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
- `./run export-voice myvoice "description..."` - Create portable voice package (zip file)
- `./run clone-voice myvoice recording.wav` - Create cloned voice from your own audio
- `./run scrape-voice "Nathan Fillion"` - Create cloned voice from celebrity
- `./run list-celebs` - List all 58 available celebrities on moviesoundclips.net
- `./run list-voices` - Show built-in, designed, and cloned voices

**Portable voice packages**:
- `./run export-voice narrator "A warm narrator..." -q hq` - Creates `narrator.voice.zip`
- Contains `voice.wav` (reference audio) and `voice.txt` (reference text)
- Uses `src/script_30.txt` (~30 seconds) for faster generation
- Quality options: `default` (fastest), `hq` (default), `ultra` (slowest)
- Use with: `./run tts "Hello" -v ./narrator.voice.zip`
- Voices are self-contained - no global registration needed

**Voice training scripts** (in `src/`):
- `src/script_30.txt` - 30-second script (~45 words) - quick voice tests
- `src/script.txt` - 2-minute phonetically balanced text (~270 words) - use for voice design
- `src/script16.txt` - 16-minute extended script (~720 words) - for maximum voice capture

## Multi-Speaker TTS

Generate dialogue with multiple voices from JSONL input.

**Command**: `./run multi dialogue.jsonl -o output.mp3`

**Input format**: JSONL with one JSON object per line, each with exactly one key (voice name) and value (text):
```jsonl
{"bob": "Hello there!"}
{"jane": "Well hello to you too."}
{"bob": "How are you today?"}
{"jane": "Doing great, thanks for asking."}
```

**Features**:
- **Fail-fast validation**: All voices are validated upfront before any synthesis begins
- **Consecutive speaker grouping**: Multiple lines from same speaker are merged (e.g., 3 consecutive bob lines become one segment)
- **Automatic silence trimming**: Leading and trailing silence is removed from each segment for tight dialogue transitions
- **Standard chunking**: Each speaker segment uses the same ~600 word chunking rules
- **Sequential generation**: Speakers are processed one at a time (never parallel - GPU safety)

**Options**:
- `-o, --output`: Output file path (default: input.mp3)
- `-l, --language`: Language (default: English)
- `-t, --temperature`: Synthesis variability (default: 0.9)
- `-s, --speed`: Speed multiplier 0.5-2.0 (default: 1.0)
- `-e, --enhance`: Apply AI enhancement (ultra quality, slower)

## Models

Models served by arbiter on spark (PyTorch, CUDA):

**CustomVoice** (preset voices): `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
**VoiceDesign** (designed voices): `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
**Voice Cloning** (Base model): `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

Local MLX equivalents (in tts_engine.py, not default):
- `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`
- `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`
- `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` / `0.6B-Base-bf16`

**Chunking at 600 words is mandatory** — model crashes at ~750 words even on CUDA with 128GB

## Gotchas

- **Arbiter must be running** on spark (10.0.0.254:8400) — managed by `~/bin/auto` on spark
- **Sample rate is 24kHz** (not 12kHz - the "12Hz" in model name is token rate)
- **MP3 output uses maximum quality** (qscale:a 0, ~245 kbps VBR)
- Audio normalization uses ffmpeg `loudnorm` and `alimiter` filters to prevent clipping
- Temperature parameter (`-t`) controls synthesis variability (default 0.9)
- **SSML not supported** - Qwen3-TTS uses natural language instructions for prosody, not XML tags
- **600-word chunk limit is model-inherent** — tested: 748 words crashes the model even on CUDA with 128GB
- Voice cloning ref_audio is SCP'd to `/tmp/arbiter-inbox/` on spark for efficiency
- Voice scraping uses **ultra quality enhancement by default** (nfe=128, slowest but best)
- **Enhancement output is 44.1kHz** - resemble-enhance upsamples from 24kHz to 44.1kHz
- **Multi-speaker silence trimming** - uses ffmpeg silenceremove filter (-50dB threshold, 0.1s min duration) to remove leading/trailing silence from each segment for tight dialogue
- **Arbiter serializes per model** — submitting 10 chunks queues them all; no local locking needed
