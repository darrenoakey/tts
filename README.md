![](banner.jpg)

Now I have a thorough understanding of the project. Here's the README:

# TTS

Text-to-speech for macOS (Apple Silicon). Convert text to natural-sounding speech with built-in voices, custom voice design, voice cloning, and multi-speaker dialogue.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)

## Installation

```bash
./run install
```

This creates a virtual environment and installs all dependencies. The first run will also download the TTS model (~5GB, cached for future use).

## Usage

### Generate Speech from a Text File

```bash
./run tts article.txt -o article.mp3
```

### Generate Speech from Inline Text

```bash
./run tts "Hello, welcome to our presentation." -o greeting.mp3
```

### Choose a Voice

```bash
./run tts "Good morning everyone." -v ryan -o morning.mp3
```

Built-in English voices: `aiden`, `ryan`, `ono_anna`, `sohee`

Built-in Chinese voices: `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`

### Adjust Speed and Temperature

```bash
# Speak faster (range: 0.5 to 2.0)
./run tts "Breaking news today." -v aiden -s 1.3 -o fast.mp3

# Lower temperature for more consistent output (default: 0.9)
./run tts "Important announcement." -t 0.5 -o announcement.mp3
```

### Apply AI Enhancement

```bash
./run tts article.txt -o enhanced.mp3 -e
```

The `-e` flag applies audio enhancement for higher quality output (significantly slower).

### List Available Voices

```bash
./run list-voices
```

## Voice Creation

### Design a Voice from a Description

Create a custom voice by describing what it should sound like:

```bash
./run design-voice narrator "A warm, confident male narrator with a deep baritone voice and measured pace"
```

Once designed, use it by name:

```bash
./run tts "Once upon a time..." -v narrator -o story.mp3
```

### Clone a Voice from Audio

Create a voice clone from your own audio recording:

```bash
./run clone-voice myvoice recording.wav
```

The audio is automatically transcribed and registered. Use it immediately:

```bash
./run tts "This sounds like me." -v myvoice -o clone_test.mp3
```

### Clone a Celebrity Voice

Create a voice from celebrity movie clips:

```bash
# See who's available
./run list-celebs

# Create the voice
./run scrape-voice "Morgan Freeman"
```

Quality options for voice scraping:

```bash
./run scrape-voice "Morgan Freeman" --hq     # High quality (faster)
./run scrape-voice "Morgan Freeman" --fast   # Fast (lowest quality)
# Default is ultra quality (slowest, best results)
```

### Export a Portable Voice Package

Create a self-contained voice file you can share or use anywhere:

```bash
./run export-voice narrator "A warm British newsreader with a clear, authoritative tone"
```

This creates `narrator.voice.zip`. Use it directly:

```bash
./run tts "Hello world" -v ./narrator.voice.zip -o hello.mp3
```

Quality options for export: `default` (fastest), `hq` (default), `ultra` (slowest):

```bash
./run export-voice narrator "A warm narrator voice" -q ultra
```

## Multi-Speaker Dialogue

Generate audio with multiple voices from a JSONL file where each line assigns text to a speaker.

Create a dialogue file (`conversation.jsonl`):

```jsonl
{"aiden": "Welcome to the show! Today we have a special guest."}
{"ryan": "Thanks for having me, it's great to be here."}
{"aiden": "So tell us about your latest project."}
{"ryan": "Well, it all started about three years ago..."}
```

Generate the audio:

```bash
./run multi conversation.jsonl -o conversation.mp3
```

You can use any combination of built-in, designed, or cloned voices. All voices are validated before synthesis begins.

Options:

```bash
./run multi dialogue.jsonl -o output.mp3 -s 1.2        # Faster speed
./run multi dialogue.jsonl -o output.mp3 -e             # AI enhancement
./run multi dialogue.jsonl -o output.mp3 -w ./progress  # Resumable (saves progress)
```

## Command Reference

| Command | Description |
|---------|-------------|
| `./run install` | Install dependencies |
| `./run tts <input> [options]` | Generate speech from text or file |
| `./run list-voices` | List all available voices |
| `./run design-voice <name> <description>` | Create a voice from a text description |
| `./run export-voice <name> <description>` | Create a portable voice package (zip) |
| `./run clone-voice <name> <audio>` | Clone a voice from an audio file |
| `./run scrape-voice <name>` | Clone a celebrity voice from movie clips |
| `./run list-celebs` | List available celebrities for scraping |
| `./run multi <jsonl> [options]` | Generate multi-speaker dialogue |

### TTS Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output file path (default: `input.mp3`) |
| `-v, --voice` | Voice name or `.voice.zip` path (default: `aiden`) |
| `-l, --language` | Language (default: `English`) |
| `-t, --temperature` | Variability, lower = more consistent (default: `0.9`) |
| `-s, --speed` | Speed multiplier from 0.5 to 2.0 (default: `1.0`) |
| `-e, --enhance` | Apply AI audio enhancement (slower) |
| `-d, --voice-description` | Design a voice on-the-fly from a description |

## License

This project is licensed under [CC BY-NC 4.0](https://darren-static.waft.dev/license) - free to use and modify, but no commercial use without permission.
