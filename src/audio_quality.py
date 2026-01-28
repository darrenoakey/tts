#!/usr/bin/env python3
"""Audio quality assessment and noise reduction for voice cloning."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf


# ##################################################################
# estimate snr
# estimate signal-to-noise ratio using voice activity detection
def estimate_snr(audio: np.ndarray, sr: int) -> float:
    # simple energy-based VAD
    frame_size = int(sr * 0.025)  # 25ms frames
    hop_size = int(sr * 0.010)    # 10ms hop

    # compute frame energies
    energies = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        energy = np.sum(frame ** 2) / frame_size
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
        frame = audio[i:i + frame_size]
        energy = np.sum(frame ** 2) / frame_size
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
        "ffmpeg", "-y", "-i", str(tmp_path),
        "-ar", str(sr_target),
        "-ac", "1",  # mono
        str(output_path)
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


def enhance_audio(input_path: Path, output_path: Path, sr_target: int = 24000) -> Path:
    if not ENHANCE_TOOL.exists():
        print(f"  Warning: {ENHANCE_TOOL} not found, falling back to basic noise reduction")
        return reduce_noise_basic(input_path, output_path, sr_target)

    # create temp file for enhanced output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_enhanced = Path(tmp.name)

    # run resemble-enhance
    cmd = [str(ENHANCE_TOOL), "enhance", str(input_path), str(tmp_enhanced)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Warning: resemble-enhance failed, falling back to basic: {result.stderr[:200]}")
        tmp_enhanced.unlink(missing_ok=True)
        return reduce_noise_basic(input_path, output_path, sr_target)

    # resample to target sample rate
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(tmp_enhanced),
        "-ar", str(sr_target),
        "-ac", "1",  # mono
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp_enhanced.unlink()

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg resample failed: {result.stderr}")

    return output_path


# default to AI enhancement
reduce_noise = enhance_audio


# ##################################################################
# select best clips
# select highest quality clips up to max duration
def select_best_clips(clip_dir: Path, max_duration: float = 15.0) -> list[dict]:
    clips = list(clip_dir.glob("clip_*.wav"))
    # exclude already processed clips
    clips = [c for c in clips if "_24k" not in c.name and "_clean" not in c.name]

    if not clips:
        return []

    # analyze all clips
    print(f"Analyzing {len(clips)} clips...")
    analyses = []
    for clip in clips:
        try:
            analysis = analyze_clip(clip)
            analyses.append(analysis)
        except Exception as e:
            print(f"  Skipping {clip.name}: {e}")

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
    clip_dir: Path,
    output_path: Path,
    max_duration: float = 15.0
) -> tuple[Path, list[dict]]:
    # select best clips
    selected = select_best_clips(clip_dir, max_duration)

    if not selected:
        raise ValueError("No clips found")

    print(f"\nSelected {len(selected)} clips ({sum(c['duration'] for c in selected):.1f}s total):")
    for c in selected:
        print(f"  {c['path'].name}: {c['duration']:.1f}s, SNR={c['snr']:.1f}dB, quality={c['quality_score']:.1f}")

    # enhance each selected clip with AI
    print("\nEnhancing clips with AI (this may take a few minutes per clip)...")
    clean_paths = []
    for i, c in enumerate(selected):
        clean_path = c["path"].parent / f"{c['path'].stem}_enhanced.wav"
        print(f"  [{i+1}/{len(selected)}] Enhancing {c['path'].name} ({c['duration']:.1f}s)...")
        reduce_noise(c["path"], clean_path)
        clean_paths.append(clean_path)

    # concatenate clean clips
    print(f"\nConcatenating to {output_path}...")
    if len(clean_paths) == 1:
        import shutil
        shutil.copy(clean_paths[0], output_path)
    else:
        # use ffmpeg concat
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_list = Path(f.name)
            for p in clean_paths:
                f.write(f"file '{p}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_path)
        ]
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
        print("Usage: python -m src.audio_quality <clip_dir> [max_duration]")
        print("  Analyzes clips in directory and prepares cleaned reference audio")
        sys.exit(1)

    clip_dir = Path(sys.argv[1])
    max_duration = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

    output_path = clip_dir / "reference_clean.wav"
    prepare_reference_audio(clip_dir, output_path, max_duration)
    print(f"\nReference audio ready: {output_path}")
