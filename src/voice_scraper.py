#!/usr/bin/env python3
"""Scrape voice samples from moviesoundclips.net and create cloned voices."""

import re
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from colorama import Fore, Style, init

from src.tts_engine import VOICE_REGISTRY_PATH

init(autoreset=True)


# ##################################################################
# transcribe audio
# use mlx-whisper to transcribe audio file
def transcribe_audio(audio_path: Path) -> str:
    import mlx_whisper
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    )
    return result["text"].strip()

BASE_URL = "https://www.moviesoundclips.net"
LOCAL_DIR = Path(__file__).parent.parent / "local" / "movie"


# ##################################################################
# list celebrities
# list all celebrities available on moviesoundclips.net
def list_celebrities() -> list[tuple[int, str]]:
    pages = ["", "?page=1", "?page=2", "?page=3"]
    celebs = []

    for page in pages:
        url = f"{BASE_URL}/people.php{page}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # extract all celebrity links
        # pattern: <a href="people-details.php?id=28" ...>Nathan Fillion</a>
        pattern = r'<a href="people-details\.php\?id=(\d+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, response.text)

        for person_id, name in matches:
            celebs.append((int(person_id), name.strip()))

    # sort by name
    celebs.sort(key=lambda x: x[1].lower())
    return celebs


# ##################################################################
# search person
# search for a person on moviesoundclips.net and return their page URL
def search_person(name: str) -> tuple[int, str] | None:
    # search all pages (A-F, G-L, M-Q, R-Z)
    pages = ["", "?page=1", "?page=2", "?page=3"]

    for page in pages:
        url = f"{BASE_URL}/people.php{page}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # look for the person's link
        # pattern: <a href="people-details.php?id=28" ...>Nathan Fillion</a>
        pattern = rf'<a href="people-details\.php\?id=(\d+)"[^>]*>([^<]*{re.escape(name)}[^<]*)</a>'
        match = re.search(pattern, response.text, re.IGNORECASE)
        if match:
            person_id = int(match.group(1))
            full_name = match.group(2).strip()
            return person_id, full_name

    return None


# ##################################################################
# get clip urls
# extract all audio clip URLs from a person's page
def get_clip_urls(person_id: int) -> list[str]:
    url = f"{BASE_URL}/people-details.php?id={person_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # extract WAV source URLs (highest quality)
    # pattern: <source src="/tv1/castle/hot.wav" type="audio/wav">
    pattern = r'<source src="(/[^"]+\.wav)" type="audio/wav">'
    matches = re.findall(pattern, response.text)

    # convert to full URLs
    return [urljoin(BASE_URL, path) for path in matches]


# ##################################################################
# download clips
# download all clips to a local directory
def download_clips(urls: list[str], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for i, url in enumerate(urls):
        filename = f"clip_{i:03d}.wav"
        output_path = output_dir / filename

        if output_path.exists():
            print(f"  {Fore.YELLOW}[{i+1}/{len(urls)}]{Style.RESET_ALL} {Style.DIM}Already exists:{Style.RESET_ALL} {Fore.CYAN}{filename}{Style.RESET_ALL}")
            downloaded.append(output_path)
            continue

        print(f"  {Fore.BLUE}[{i+1}/{len(urls)}]{Style.RESET_ALL} Downloading: {Fore.CYAN}{url.split('/')[-1]}{Style.RESET_ALL}")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            downloaded.append(output_path)
            time.sleep(0.5)  # be nice to the server
        except Exception as e:
            print(f"    {Fore.RED}✗ Failed:{Style.RESET_ALL} {e}")

    return downloaded


# ##################################################################
# concatenate with silence
# concatenate audio files with silence between them
def concatenate_with_silence(wav_files: list[Path], output_path: Path, silence_ms: int = 500) -> Path:
    if not wav_files:
        raise ValueError("No audio files to concatenate")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # create a concat file list with silence between clips
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_list = Path(f.name)
        for i, wav_path in enumerate(wav_files):
            f.write(f"file '{wav_path}'\n")
            # silence will be added via ffmpeg filter

    # use ffmpeg to concatenate with silence
    # first, create a filter complex that adds silence between clips
    filter_parts = []
    inputs = []
    for i, wav_path in enumerate(wav_files):
        inputs.extend(["-i", str(wav_path)])
        filter_parts.append(f"[{i}:a]aresample=24000[a{i}]")

    # build the concat filter with silence
    concat_inputs = "".join(f"[a{i}]" for i in range(len(wav_files)))
    filter_str = ";".join(filter_parts) + f";{concat_inputs}concat=n={len(wav_files)}:v=0:a=1[out]"

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", filter_str,
        "-map", "[out]",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    concat_list.unlink()

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

    return output_path


# ##################################################################
# create voice from person
# scrape clips and create a cloned voice
def create_voice_from_person(name: str, speed: float = 1.0, quality: str = "ultra") -> str:
    print(f"\n{Fore.BLUE}🔍 Searching for '{Fore.CYAN}{name}{Fore.BLUE}'...{Style.RESET_ALL}")
    result = search_person(name)
    if not result:
        raise ValueError(f"Could not find '{name}' on moviesoundclips.net")

    person_id, full_name = result
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Found: {Fore.CYAN}{Style.BRIGHT}{full_name}{Style.RESET_ALL} {Style.DIM}(id={person_id}){Style.RESET_ALL}")

    # create voice name from full name
    voice_name = full_name.lower().replace(" ", "_").replace(".", "").replace("'", "")

    # create output directory
    output_dir = LOCAL_DIR / voice_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # get clip URLs
    print(f"\n{Fore.BLUE}📋 Getting clip URLs...{Style.RESET_ALL}")
    urls = get_clip_urls(person_id)
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Found {Fore.YELLOW}{len(urls)}{Style.RESET_ALL} clips")

    if not urls:
        raise ValueError(f"No clips found for {full_name}")

    # download clips
    print(f"\n{Fore.BLUE}⬇️  Downloading clips...{Style.RESET_ALL}")
    wav_files = download_clips(urls, output_dir)
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Downloaded {Fore.YELLOW}{len(wav_files)}{Style.RESET_ALL} clips")

    # select best clips and clean them (300 seconds of best quality audio)
    from src.audio_quality import prepare_reference_audio
    reference_path = output_dir / "reference_clean.wav"
    quality_note = f", {quality} mode" if quality != "default" else ""
    print(f"\n{Fore.BLUE}🎵 Preparing clean reference audio{Style.RESET_ALL} {Style.DIM}(selecting best 300s{quality_note})...{Style.RESET_ALL}")
    prepare_reference_audio(output_dir, reference_path, max_duration=300.0, quality=quality)

    # transcribe the reference audio for voice cloning
    print(f"\n{Fore.BLUE}📝 Transcribing reference audio with Whisper...{Style.RESET_ALL}")
    ref_text = transcribe_audio(reference_path)
    transcription_preview = ref_text[:100] + "..." if len(ref_text) > 100 else ref_text
    print(f"{Fore.GREEN}✓{Style.RESET_ALL} Transcription: {Style.DIM}{transcription_preview}{Style.RESET_ALL}")

    # register the voice with a reference to the audio file and transcription
    # voice cloning requires both ref_audio and ref_text
    registry_data = {
        "type": "clone",
        "ref_audio": str(reference_path),
        "ref_text": ref_text,
        "speed": speed,
    }

    # save to voice registry
    import json
    registry = {}
    if VOICE_REGISTRY_PATH.exists():
        registry = json.loads(VOICE_REGISTRY_PATH.read_text())
    registry[voice_name] = registry_data
    VOICE_REGISTRY_PATH.write_text(json.dumps(registry, indent=2))

    print(f"\n{Fore.GREEN}✓{Style.RESET_ALL} Registered voice '{Fore.CYAN}{Style.BRIGHT}{voice_name}{Style.RESET_ALL}'")
    return voice_name


# ##################################################################
# main
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"{Fore.YELLOW}Usage:{Style.RESET_ALL} python -m src.voice_scraper {Fore.CYAN}<person_name>{Style.RESET_ALL} [--hq|--fast]")
        print(f"  {Style.DIM}Default: Ultra quality (nfe=256, slowest but best){Style.RESET_ALL}")
        print(f"  {Style.DIM}--hq: High quality (nfe=128, faster){Style.RESET_ALL}")
        print(f"  {Style.DIM}--fast: Fast mode (nfe=64, fastest){Style.RESET_ALL}")
        sys.exit(1)

    # parse quality flag (default is ultra)
    quality = "ultra"
    args = sys.argv[1:]
    if "--fast" in args:
        quality = "default"
        args.remove("--fast")
    elif "--hq" in args:
        quality = "hq"
        args.remove("--hq")

    name = " ".join(args)
    voice_name = create_voice_from_person(name, quality=quality)
    print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 Voice '{voice_name}' is ready!{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Use with:{Style.RESET_ALL} {Fore.CYAN}./run tts input.txt -v {voice_name}{Style.RESET_ALL}\n")
