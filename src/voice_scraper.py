#!/usr/bin/env python3
"""Scrape voice samples from moviesoundclips.net and create cloned voices."""

import re
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

from src.tts_engine import VOICE_REGISTRY_PATH


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
            print(f"  [{i+1}/{len(urls)}] Already exists: {filename}")
            downloaded.append(output_path)
            continue

        print(f"  [{i+1}/{len(urls)}] Downloading: {url}")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            output_path.write_bytes(response.content)
            downloaded.append(output_path)
            time.sleep(0.5)  # be nice to the server
        except Exception as e:
            print(f"    Failed: {e}")

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
def create_voice_from_person(name: str, speed: float = 1.0) -> str:
    print(f"Searching for '{name}'...")
    result = search_person(name)
    if not result:
        raise ValueError(f"Could not find '{name}' on moviesoundclips.net")

    person_id, full_name = result
    print(f"Found: {full_name} (id={person_id})")

    # create voice name from full name
    voice_name = full_name.lower().replace(" ", "_").replace(".", "").replace("'", "")

    # create output directory
    output_dir = LOCAL_DIR / voice_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # get clip URLs
    print("Getting clip URLs...")
    urls = get_clip_urls(person_id)
    print(f"Found {len(urls)} clips")

    if not urls:
        raise ValueError(f"No clips found for {full_name}")

    # download clips
    print("Downloading clips...")
    wav_files = download_clips(urls, output_dir)
    print(f"Downloaded {len(wav_files)} clips")

    # select best clips and clean them (40 seconds of best quality audio)
    from src.audio_quality import prepare_reference_audio
    reference_path = output_dir / "reference_clean.wav"
    print("\nPreparing clean reference audio (selecting best 40s of clips, removing noise)...")
    prepare_reference_audio(output_dir, reference_path, max_duration=40.0)

    # transcribe the reference audio for voice cloning
    print("\nTranscribing reference audio with Whisper...")
    ref_text = transcribe_audio(reference_path)
    print(f"Transcription: {ref_text[:100]}..." if len(ref_text) > 100 else f"Transcription: {ref_text}")

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

    print(f"Registered voice '{voice_name}'")
    return voice_name


# ##################################################################
# main
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.voice_scraper <person_name>")
        sys.exit(1)

    name = " ".join(sys.argv[1:])
    voice_name = create_voice_from_person(name)
    print(f"\nVoice '{voice_name}' is ready!")
    print(f"Use with: ./run tts input.txt -v {voice_name}")
