#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import setproctitle
from colorama import Fore, Style, init

from src.tts_engine import QWEN_VOICES, DEFAULT_VOICE, DEFAULT_TEMPERATURE, DEFAULT_SPEED, register_voice, list_custom_voices

init(autoreset=True)


# ##################################################################
# main
# entry point for tts command - converts text files to speech audio
def main(argv: list[str]) -> int:
    setproctitle.setproctitle("tts")

    custom_voices = list_custom_voices()
    voice_help = f"Voice to use (default: {DEFAULT_VOICE}). Built-in: {', '.join(QWEN_VOICES)}"
    if custom_voices:
        voice_help += f". Custom: {', '.join(custom_voices)}"

    parser = argparse.ArgumentParser(
        prog="tts",
        description="Convert text files to speech audio",
    )
    sub = parser.add_subparsers(dest="command")

    # generate command (default)
    p_gen = sub.add_parser("generate", help="Generate speech from text or file")
    p_gen.add_argument(
        "input",
        type=str,
        help="Input text file or inline text (e.g., 'hello world')",
    )
    p_gen.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output audio file (default: input.mp3 or output.mp3 for inline text)",
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
    sub.add_parser("list-voices", help="List available voices")

    args = parser.parse_args(argv)

    # default to generate if no subcommand but has positional args
    if args.command is None:
        # re-parse with generate as implicit command
        return main(["generate"] + argv)

    if args.command == "register-voice":
        register_voice(args.name, args.description, args.speed)
        print(f"{Fore.GREEN}✓{Style.RESET_ALL} Registered voice '{Fore.CYAN}{args.name}{Style.RESET_ALL}' with speed {Fore.YELLOW}{args.speed}{Style.RESET_ALL}")
        return 0

    if args.command == "list-voices":
        print(f"\n{Fore.BLUE}{Style.BRIGHT}Built-in voices:{Style.RESET_ALL}")
        for v in QWEN_VOICES:
            print(f"  {Fore.WHITE}•{Style.RESET_ALL} {Fore.CYAN}{v}{Style.RESET_ALL}")
        custom = list_custom_voices()
        if custom:
            from src.tts_engine import load_voice_registry
            registry = load_voice_registry()
            designed = []
            cloned = []
            for v in custom:
                info = registry[v]
                if info.get("type") == "clone":
                    cloned.append((v, info))
                else:
                    designed.append((v, info))
            if designed:
                print(f"\n{Fore.MAGENTA}{Style.BRIGHT}Designed voices:{Style.RESET_ALL}")
                for v, info in designed:
                    desc = info.get("description", "No description")
                    speed = info.get('speed', 1.0)
                    print(f"  {Fore.WHITE}•{Style.RESET_ALL} {Fore.GREEN}{v}{Style.RESET_ALL}: {Fore.WHITE}{desc[:50]}...{Style.RESET_ALL} {Style.DIM}(speed: {speed}){Style.RESET_ALL}")
            if cloned:
                print(f"\n{Fore.YELLOW}{Style.BRIGHT}Cloned voices:{Style.RESET_ALL}")
                for v, info in cloned:
                    ref_audio = info.get("ref_audio", "unknown")
                    speed = info.get('speed', 1.0)
                    print(f"  {Fore.WHITE}•{Style.RESET_ALL} {Fore.GREEN}{v}{Style.RESET_ALL}: ref={Fore.CYAN}{Path(ref_audio).name}{Style.RESET_ALL} {Style.DIM}(speed: {speed}){Style.RESET_ALL}")
        print()
        return 0

    if args.command == "generate":
        try:
            from src.convert import convert_text_to_speech
            # check if input is a file or inline text
            input_path = Path(args.input)
            if input_path.exists():
                # it's a file
                output_path = convert_text_to_speech(
                    input_path=input_path,
                    output_path=args.output,
                    model=args.model,
                    language=args.language,
                    voice=args.voice,
                    temperature=args.temperature,
                    speed=args.speed,
                    voice_description=args.voice_description,
                )
            else:
                # treat as inline text
                output_path = convert_text_to_speech(
                    text=str(args.input),
                    output_path=args.output,
                    model=args.model,
                    language=args.language,
                    voice=args.voice,
                    temperature=args.temperature,
                    speed=args.speed,
                    voice_description=args.voice_description,
                )
            print(f"{Fore.GREEN}✓{Style.RESET_ALL} Generated: {Fore.CYAN}{output_path}{Style.RESET_ALL}")
            return 0
        except FileNotFoundError as err:
            print(f"{Fore.RED}✗ Error:{Style.RESET_ALL} {err}", file=sys.stderr)
            return 1
        except ValueError as err:
            print(f"{Fore.RED}✗ Error:{Style.RESET_ALL} {err}", file=sys.stderr)
            return 1
        except Exception as err:
            print(f"{Fore.RED}✗ Error:{Style.RESET_ALL} {err}", file=sys.stderr)
            return 1

    return 0


# ##################################################################
# entry point
# standard python dispatch to main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
