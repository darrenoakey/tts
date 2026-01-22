#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import setproctitle

from src.tts_engine import QWEN_VOICES, DEFAULT_VOICE


# ##################################################################
# main
# entry point for tts command - converts text files to speech audio
def main(argv: list[str]) -> int:
    setproctitle.setproctitle("tts")

    voice_help = f"Voice to use (default: {DEFAULT_VOICE}). Available: {', '.join(QWEN_VOICES)}"

    parser = argparse.ArgumentParser(
        prog="tts",
        description="Convert text files to speech audio",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input text file to convert",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output audio file (default: input with .mp3 extension)",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="qwen",
        choices=["qwen"],
        help="TTS model to use (default: qwen)",
    )
    parser.add_argument(
        "-v", "--voice",
        type=str,
        default=DEFAULT_VOICE,
        choices=QWEN_VOICES,
        help=voice_help,
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default="English",
        help="Language for synthesis (default: English)",
    )

    args = parser.parse_args(argv)

    try:
        from src.convert import convert_text_to_speech
        output_path = convert_text_to_speech(
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            language=args.language,
            voice=args.voice,
        )
        print(f"Generated: {output_path}")
        return 0
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1


# ##################################################################
# entry point
# standard python dispatch to main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
