from pathlib import Path
from src.tts_engine import get_engine, get_voice_description, DEFAULT_VOICE, DEFAULT_TEMPERATURE, DEFAULT_SPEED


# ##################################################################
# convert text to speech
# read text from input file and synthesize to audio output file
def convert_text_to_speech(
    input_path: Path,
    output_path: Path | None = None,
    model: str = "qwen",
    language: str = "English",
    voice: str = DEFAULT_VOICE,
    temperature: float = DEFAULT_TEMPERATURE,
    speed: float | None = None,
    voice_description: str | None = None,
) -> Path:
    # ##################################################################
    # convert text to speech
    # orchestrates reading input, getting engine, and writing audio output
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Input file is empty: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".mp3")
    else:
        output_path = Path(output_path)

    # check if voice is a custom registered voice with its own speed
    custom_voice = get_voice_description(voice)
    if speed is None:
        speed = custom_voice.get("speed", DEFAULT_SPEED) if custom_voice else DEFAULT_SPEED

    engine = get_engine(model, voice=voice, voice_description=voice_description)
    return engine.synthesize(text, output_path, language=language, temperature=temperature, speed=speed)


# ##################################################################
# convert module
# provides the high-level text-to-speech conversion function that
# reads text files and produces audio output using configured engines
