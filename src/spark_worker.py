#!/usr/bin/env python3
"""TTS worker for spark (NVIDIA CUDA).

Reads a JSON config file, loads the appropriate Qwen3-TTS model,
synthesizes audio for all text chunks, and writes a single combined WAV.

Designed to run on the spark machine with PyTorch + CUDA.
"""

import gc
import json
import sys

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def load_model(model_name, try_flash_attn=True):
    """Load a Qwen3-TTS model with optional flash attention."""
    kwargs = {
        "device_map": "cuda:0",
        "dtype": torch.bfloat16,
    }
    if try_flash_attn:
        try:
            kwargs["attn_implementation"] = "flash_attention_2"
            return Qwen3TTSModel.from_pretrained(model_name, **kwargs)
        except Exception:
            # flash-attn not installed or incompatible, fall back
            del kwargs["attn_implementation"]
    return Qwen3TTSModel.from_pretrained(model_name, **kwargs)


def main():
    if len(sys.argv) != 2:
        print("Usage: worker.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config = json.loads(open(sys.argv[1]).read())

    mode = config["mode"]  # "custom", "clone", "design"
    chunks = config["chunks"]
    output_path = config["output"]
    language = config.get("language", "English")
    temperature = config.get("temperature", 0.9)

    print(f"Mode: {mode}, Chunks: {len(chunks)}, Language: {language}", flush=True)

    # load model
    if mode == "custom":
        model_name = config.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        print(f"Loading model: {model_name}", flush=True)
        model = load_model(model_name)
    elif mode == "clone":
        model_name = config.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        print(f"Loading model: {model_name}", flush=True)
        model = load_model(model_name)
        # create reusable voice clone prompt (encodes ref audio once)
        ref_audio = config["ref_audio"]
        ref_text = config.get("ref_text")
        print(f"Creating voice clone prompt from: {ref_audio}", flush=True)
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
    elif mode == "design":
        model_name = config.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        print(f"Loading model: {model_name}", flush=True)
        model = load_model(model_name)
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    print("Model loaded, starting synthesis...", flush=True)

    all_audio = []
    sample_rate = 24000  # qwen3-tts output is always 24kHz

    voice_clone_prompt = locals().get("voice_clone_prompt")

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...", flush=True)

        if mode == "custom":
            wavs, sample_rate = model.generate_custom_voice(
                text=chunk,
                language=language,
                speaker=config.get("voice", "Aiden"),
                temperature=temperature,
            )
        elif mode == "clone":
            wavs, sample_rate = model.generate_voice_clone(
                text=chunk,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                temperature=temperature,
            )
        elif mode == "design":
            wavs, sample_rate = model.generate_voice_design(
                text=chunk,
                language=language,
                instruct=config.get("voice_description", "A clear neutral voice."),
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        all_audio.append(wavs[0])
        print(f"Chunk {i + 1}/{len(chunks)} complete", flush=True)

        # memory cleanup between chunks
        gc.collect()
        torch.cuda.empty_cache()

    # concatenate all audio and write output
    combined = np.concatenate(all_audio)
    sf.write(output_path, combined, sample_rate)
    print(f"Output written: {output_path} ({len(combined) / sample_rate:.1f}s, {sample_rate}Hz)", flush=True)


def main_batch():
    """Batch mode: process multiple output files from a single model load.

    Config format for batch mode:
    {
        "mode": "custom",
        "batch": [
            {"chunks": ["text1"], "output": "/path/out1.wav"},
            {"chunks": ["text2", "text3"], "output": "/path/out2.wav"},
        ],
        "language": "English",
        "temperature": 0.9,
        ...
    }
    """
    config = json.loads(open(sys.argv[1]).read())

    mode = config["mode"]
    batch = config["batch"]
    language = config.get("language", "English")
    temperature = config.get("temperature", 0.9)

    total_items = len(batch)
    print(f"Batch mode: {mode}, Items: {total_items}, Language: {language}", flush=True)

    # load model once
    if mode == "custom":
        model_name = config.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        print(f"Loading model: {model_name}", flush=True)
        model = load_model(model_name)
    elif mode == "clone":
        model_name = config.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        print(f"Loading model: {model_name}", flush=True)
        model = load_model(model_name)
        ref_audio = config["ref_audio"]
        ref_text = config.get("ref_text")
        print(f"Creating voice clone prompt from: {ref_audio}", flush=True)
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
    elif mode == "design":
        model_name = config.get("model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        print(f"Loading model: {model_name}", flush=True)
        model = load_model(model_name)
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)

    print("Model loaded, starting batch synthesis...", flush=True)
    voice_clone_prompt_ref = locals().get("voice_clone_prompt")

    for item_idx, item in enumerate(batch):
        chunks = item["chunks"]
        output_path = item["output"]
        all_audio = []
        sample_rate = 24000

        print(f"\nItem {item_idx + 1}/{total_items}: {len(chunks)} chunk(s) -> {output_path}", flush=True)

        for i, chunk in enumerate(chunks):
            if mode == "custom":
                wavs, sample_rate = model.generate_custom_voice(
                    text=chunk,
                    language=language,
                    speaker=config.get("voice", "Aiden"),
                    temperature=temperature,
                )
            elif mode == "clone":
                wavs, sample_rate = model.generate_voice_clone(
                    text=chunk,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt_ref,
                    temperature=temperature,
                )
            elif mode == "design":
                wavs, sample_rate = model.generate_voice_design(
                    text=chunk,
                    language=language,
                    instruct=config.get("voice_description", "A clear neutral voice."),
                    temperature=temperature,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            all_audio.append(wavs[0])
            gc.collect()
            torch.cuda.empty_cache()

        combined = np.concatenate(all_audio)
        sf.write(output_path, combined, sample_rate)
        print(f"Item {item_idx + 1}/{total_items} written ({len(combined) / sample_rate:.1f}s)", flush=True)

    print(f"\nBatch complete: {total_items} items", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] == "--batch":
        main_batch()
    else:
        main()
