#!/usr/bin/env python3
"""Audio quality assessment and noise reduction for voice cloning."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf
from colorama import Fore, Style, init

init(autoreset=True)


# ##################################################################
# estimate snr
# estimate signal-to-noise ratio using voice activity detection
def estimate_snr(audio: np.ndarray, sr: int) -> float:
    # simple energy-based VAD
    frame_size = int(sr * 0.025)  # 25ms frames
    hop_size = int(sr * 0.010)  # 10ms hop

    # compute frame energies
    energies = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i : i + frame_size]
        energy = np.sum(frame**2) / frame_size
        energies.append(energy)

    if not energies:
        return 0.0

    energies = np.array(energies)

    # threshold for voice activity (adaptive)
    threshold = np.percentile(energies, 30)

    # separate speech and noise frames
    speech_energies = energies[energies > threshold]
    noise_energies = energies[energies <= threshold]

    if len(noise_energies) == 0 or len(speech_energies) == 0:
        return 20.0  # assume good quality if can't estimate

    speech_power = np.mean(speech_energies)
    noise_power = np.mean(noise_energies)

    if noise_power == 0:
        return 40.0  # very clean

    snr = 10 * np.log10(speech_power / noise_power)
    return snr


# ##################################################################
# estimate speech ratio
# estimate how much of the audio contains actual speech
def estimate_speech_ratio(audio: np.ndarray, sr: int) -> float:
    frame_size = int(sr * 0.025)
    hop_size = int(sr * 0.010)

    energies = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i : i + frame_size]
        energy = np.sum(frame**2) / frame_size
        energies.append(energy)

    if not energies:
        return 0.0

    energies = np.array(energies)
    threshold = np.percentile(energies, 30)

    speech_frames = np.sum(energies > threshold)
    return speech_frames / len(energies)


# ##################################################################
# analyze clip
# compute quality metrics for an audio clip
def analyze_clip(audio_path: Path) -> dict:
    audio, sr = sf.read(audio_path)

    # convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    duration = len(audio) / sr
    snr = estimate_snr(audio, sr)
    speech_ratio = estimate_speech_ratio(audio, sr)

    # compute overall quality score (higher is better)
    # weight SNR heavily, but also favor clips with more speech
    quality_score = snr * 0.7 + speech_ratio * 30 * 0.3

    return {
        "path": audio_path,
        "duration": duration,
        "snr": snr,
        "speech_ratio": speech_ratio,
        "quality_score": quality_score,
    }


# ##################################################################
# reduce noise (basic)
# apply basic noise reduction using noisereduce library
def reduce_noise_basic(input_path: Path, output_path: Path, sr_target: int = 24000) -> Path:
    audio, sr = sf.read(input_path)

    # convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # apply noise reduction
    reduced = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8, stationary=False)

    # normalize to prevent clipping
    max_val = np.max(np.abs(reduced))
    if max_val > 0:
        reduced = reduced / max_val * 0.9

    # save to temp file first, then resample with ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    sf.write(str(tmp_path), reduced, sr)

    # resample to target sample rate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp_path),
        "-ar",
        str(sr_target),
        "-ac",
        "1",  # mono
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp_path.unlink()

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed: {result.stderr}")

    return output_path


# ##################################################################
# enhance audio (AI-based)
# use resemble-enhance for high-quality audio enhancement
ENHANCE_TOOL = Path.home() / "src" / "audio-enhance" / "run"


def enhance_audio(input_path: Path, output_path: Path, sr_target: int = 24000, quality: str = "default") -> Path:
    """Enhance audio using resemble-enhance.

    Args:
        input_path: Input audio file
        output_path: Output audio file
        sr_target: Target sample rate (default 24000)
        quality: Quality mode - "default", "hq" (nfe=128), or "ultra" (nfe=256)
    """
    if not ENHANCE_TOOL.exists():
        raise RuntimeError(f"Enhancement tool not found: {ENHANCE_TOOL}")

    # create temp file for enhanced output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_enhanced = Path(tmp.name)

    # run resemble-enhance (does both denoising and enhancement)
    cmd = [str(ENHANCE_TOOL), "enhance", str(input_path), str(tmp_enhanced)]
    if quality == "hq":
        cmd.append("--hq")
    elif quality == "ultra":
        cmd.append("--ultra")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        tmp_enhanced.unlink(missing_ok=True)
        raise RuntimeError(f"Enhancement failed: {result.stderr}")

    # resample to target sample rate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp_enhanced),
        "-ar",
        str(sr_target),
        "-ac",
        "1",  # mono
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp_enhanced.unlink()

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed: {result.stderr}")

    return output_path


# ##################################################################
# select best clips
# select highest quality clips up to max duration
def select_best_clips(clip_dir: Path, max_duration: float = 15.0) -> list[dict]:
    clips = list(clip_dir.glob("clip_*.wav"))
    # exclude already processed clips (only analyze original clips)
    clips = [c for c in clips if "_24k" not in c.name and "_clean" not in c.name and "_enhanced" not in c.name]

    if not clips:
        return []

    # analyze all clips
    print(f"{Fore.BLUE}🔬 Analyzing {Fore.YELLOW}{len(clips)}{Fore.BLUE} clips...{Style.RESET_ALL}")
    analyses = []
    for clip in clips:
        try:
            analysis = analyze_clip(clip)
            analyses.append(analysis)
        except Exception as e:
            print(f"  {Fore.RED}✗{Style.RESET_ALL} Skipping {Fore.CYAN}{clip.name}{Style.RESET_ALL}: {e}")

    # sort by quality score (descending)
    analyses.sort(key=lambda x: x["quality_score"], reverse=True)

    # select best clips up to max duration
    selected = []
    total_duration = 0.0

    for analysis in analyses:
        if total_duration + analysis["duration"] <= max_duration:
            selected.append(analysis)
            total_duration += analysis["duration"]

    return selected


# ##################################################################
# prepare reference audio
# select best clips, clean them, concatenate into reference audio
def prepare_reference_audio(
    clip_dir: Path, output_path: Path, max_duration: float = 15.0, quality: str = "ultra"
) -> tuple[Path, list[dict]]:
    # select best clips
    selected = select_best_clips(clip_dir, max_duration)

    if not selected:
        raise ValueError("No clips found")

    total_duration = sum(c["duration"] for c in selected)
    print(
        f"\n{Fore.GREEN}✓{Style.RESET_ALL} Selected {Fore.YELLOW}{len(selected)}{Style.RESET_ALL} clips ({Fore.CYAN}{total_duration:.1f}s{Style.RESET_ALL} total):"
    )
    for c in selected:
        print(
            f"  {Fore.WHITE}•{Style.RESET_ALL} {Fore.CYAN}{c['path'].name}{Style.RESET_ALL}: {c['duration']:.1f}s, {Style.DIM}SNR={c['snr']:.1f}dB, quality={c['quality_score']:.1f}{Style.RESET_ALL}"
        )

    # enhance each selected clip with AI (resemble-enhance does denoising + enhancement)
    # skip clips that are already enhanced (idempotent)
    quality_label = {"default": "", "hq": " (HQ mode)", "ultra": " (ULTRA mode)"}
    print(
        f"\n{Fore.MAGENTA}🤖 Enhancing clips with AI{quality_label.get(quality, '')}{Style.RESET_ALL} {Style.DIM}(this may take a few minutes per clip)...{Style.RESET_ALL}"
    )
    clean_paths = []
    for i, c in enumerate(selected):
        clean_path = c["path"].parent / f"{c['path'].stem}_enhanced.wav"
        if clean_path.exists():
            print(
                f"  {Fore.YELLOW}[{i + 1}/{len(selected)}]{Style.RESET_ALL} {Style.DIM}Already enhanced:{Style.RESET_ALL} {Fore.CYAN}{clean_path.name}{Style.RESET_ALL}"
            )
        else:
            print(
                f"  {Fore.BLUE}[{i + 1}/{len(selected)}]{Style.RESET_ALL} Enhancing {Fore.CYAN}{c['path'].name}{Style.RESET_ALL} ({c['duration']:.1f}s)..."
            )
            enhance_audio(c["path"], clean_path, quality=quality)
            print(f"  {Fore.GREEN}✓{Style.RESET_ALL} Enhanced {Fore.CYAN}{clean_path.name}{Style.RESET_ALL}")
        clean_paths.append(clean_path)

    # concatenate clean clips
    print(f"\n{Fore.BLUE}🔗 Concatenating to {Fore.CYAN}{output_path.name}{Fore.BLUE}...{Style.RESET_ALL}")
    if len(clean_paths) == 1:
        import shutil

        shutil.copy(clean_paths[0], output_path)
    else:
        # use ffmpeg concat
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_list = Path(f.name)
            for p in clean_paths:
                f.write(f"file '{p}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        concat_list.unlink()

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")

    return output_path, selected


# ##################################################################
# main
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            f"{Fore.YELLOW}Usage:{Style.RESET_ALL} python -m src.audio_quality {Fore.CYAN}<clip_dir>{Style.RESET_ALL} [max_duration] [quality]"
        )
        print(f"  {Style.DIM}Analyzes clips in directory and prepares cleaned reference audio{Style.RESET_ALL}")
        print(f"  {Style.DIM}quality: default, hq, or ultra{Style.RESET_ALL}")
        sys.exit(1)

    clip_dir = Path(sys.argv[1])
    max_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    quality = sys.argv[3] if len(sys.argv) > 3 else "ultra"

    output_path = clip_dir / "reference_clean.wav"
    prepare_reference_audio(clip_dir, output_path, max_duration, quality=quality)
    print(f"\n{Fore.GREEN}✓{Style.RESET_ALL} Reference audio ready: {Fore.CYAN}{output_path}{Style.RESET_ALL}")
