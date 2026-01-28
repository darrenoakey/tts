from pathlib import Path
from src.tts_engine import get_engine, DEFAULT_VOICE, DEFAULT_TEMPERATURE


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

    engine = get_engine(model, voice=voice)
    return engine.synthesize(text, output_path, language=language, temperature=temperature)


# ##################################################################
# convert module
# provides the high-level text-to-speech conversion function that
# reads text files and produces audio output using configured engines
