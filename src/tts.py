#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import setproctitle

from src.tts_engine import QWEN_VOICES, DEFAULT_VOICE, DEFAULT_TEMPERATURE, DEFAULT_SPEED, register_voice, list_custom_voices


# ##################################################################
# main
# entry point for tts command - converts text files to speech audio
def main(argv: list[str]) -> int:
    setproctitle.setproctitle("tts")

    custom_voices = list_custom_voices()
    all_voices = QWEN_VOICES + custom_voices
    voice_help = f"Voice to use (default: {DEFAULT_VOICE}). Built-in: {', '.join(QWEN_VOICES)}"
    if custom_voices:
        voice_help += f". Custom: {', '.join(custom_voices)}"

    parser = argparse.ArgumentParser(
        prog="tts",
        description="Convert text files to speech audio",
    )
    sub = parser.add_subparsers(dest="command")

    # generate command (default)
    p_gen = sub.add_parser("generate", help="Generate speech from text file")
    p_gen.add_argument(
        "input",
        type=Path,
        help="Input text file to convert",
    )
    p_gen.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output audio file (default: input with .mp3 extension)",
    )
    p_gen.add_argument(
        "-m", "--model",
        type=str,
        default="qwen",
        choices=["qwen"],
        help="TTS model to use (default: qwen)",
    )
    p_gen.add_argument(
        "-v", "--voice",
        type=str,
        default=DEFAULT_VOICE,
        help=voice_help,
    )
    p_gen.add_argument(
        "-l", "--language",
        type=str,
        default="English",
        help="Language for synthesis (default: English)",
    )
    p_gen.add_argument(
        "-t", "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature for synthesis variability (default: {DEFAULT_TEMPERATURE})",
    )
    p_gen.add_argument(
        "-s", "--speed",
        type=float,
        default=None,
        help=f"Speed multiplier 0.5-2.0 (default: {DEFAULT_SPEED}, or voice's saved speed)",
    )
    p_gen.add_argument(
        "-d", "--voice-description",
        type=str,
        default=None,
        help="Voice description for voice design (creates voice on-the-fly)",
    )

    # register-voice command
    p_reg = sub.add_parser("register-voice", help="Register a custom voice description")
    p_reg.add_argument("name", help="Name for the voice (e.g., 'bob')")
    p_reg.add_argument("description", help="Voice description (e.g., 'A deep voiced man...')")
    p_reg.add_argument(
        "-s", "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=f"Default speed for this voice (default: {DEFAULT_SPEED})",
    )

    # list-voices command
    p_list = sub.add_parser("list-voices", help="List available voices")

    args = parser.parse_args(argv)

    # default to generate if no subcommand but has positional args
    if args.command is None:
        # re-parse with generate as implicit command
        return main(["generate"] + argv)

    if args.command == "register-voice":
        register_voice(args.name, args.description, args.speed)
        print(f"Registered voice '{args.name}' with speed {args.speed}")
        return 0

    if args.command == "list-voices":
        print("Built-in voices:")
        for v in QWEN_VOICES:
            print(f"  {v}")
        custom = list_custom_voices()
        if custom:
            print("\nCustom voices:")
            from src.tts_engine import load_voice_registry
            registry = load_voice_registry()
            for v in custom:
                info = registry[v]
                print(f"  {v}: {info['description'][:50]}... (speed: {info['speed']})")
        return 0

    if args.command == "generate":
        try:
            from src.convert import convert_text_to_speech
            output_path = convert_text_to_speech(
                input_path=args.input,
                output_path=args.output,
                model=args.model,
                language=args.language,
                voice=args.voice,
                temperature=args.temperature,
                speed=args.speed,
                voice_description=args.voice_description,
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

    return 0


# ##################################################################
# entry point
# standard python dispatch to main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
