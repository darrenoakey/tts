import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

# suppress transformers warnings about model type and tokenizer regex
# must be configured before transformers is imported by mlx_audio
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="transformers")


# available voices for qwen3-tts customvoice model (mlx-audio)
# english: aiden, ryan, ono_anna, sohee
# chinese: vivian, serena, uncle_fu, dylan (beijing dialect), eric (sichuan dialect)
QWEN_VOICES = ["aiden", "ryan", "ono_anna", "sohee", "vivian", "serena", "uncle_fu", "dylan", "eric"]
DEFAULT_VOICE = "aiden"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_SPEED = 1.0

# chunk size for processing long texts (in words)
# target ~600 words per chunk, stopping at sentence boundaries
# this typically results in ~500-600 word chunks
DEFAULT_CHUNK_WORDS = 600

# voice registry location
VOICE_REGISTRY_PATH = Path(__file__).parent.parent / "voices.json"


# ##################################################################
# voice registry
# save and load custom voice descriptions by name
def load_voice_registry() -> dict:
    if VOICE_REGISTRY_PATH.exists():
        return json.loads(VOICE_REGISTRY_PATH.read_text())
    return {}


def save_voice_registry(registry: dict) -> None:
    VOICE_REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def register_voice(name: str, description: str, speed: float = DEFAULT_SPEED) -> None:
    registry = load_voice_registry()
    registry[name.lower()] = {"description": description, "speed": speed}
    save_voice_registry(registry)


def get_voice_description(name: str) -> dict | None:
    registry = load_voice_registry()
    return registry.get(name.lower())


def list_custom_voices() -> list[str]:
    return list(load_voice_registry().keys())


# ##################################################################
# split text into chunks
# splits text into chunks of approximately n words, stopping at sentence boundaries
def split_into_chunks(text: str, max_words: int = DEFAULT_CHUNK_WORDS) -> list[str]:
    text = text.strip()
    if not text:
        return []

    # split on sentence boundaries (., !, ?) followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # if total words <= max_words, return as single chunk
    total_words = len(text.split())
    if total_words <= max_words:
        return [text]

    chunks = []
    current_chunk: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        # if adding this sentence would exceed limit and we have content, start new chunk
        if current_word_count + sentence_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        current_chunk.append(sentence)
        current_word_count += sentence_words

    # add remaining content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ##################################################################
# convert wav to mp3
# use ffmpeg to compress wav audio to mp3 format with audio normalization and speed
def convert_wav_to_mp3(wav_path: Path, mp3_path: Path, normalize: bool = True, speed: float = 1.0) -> Path:
    filters = []

    # speed adjustment (atempo only supports 0.5-2.0, chain for more extreme)
    if speed != 1.0:
        speed = max(0.5, min(2.0, speed))  # clamp to valid range
        filters.append(f"atempo={speed}")

    if normalize:
        # use ffmpeg's loudnorm filter for EBU R128 loudness normalization
        # and alimiter to prevent clipping - very conservative settings
        filters.append("loudnorm=I=-28:TP=-9:LRA=7")
        filters.append("alimiter=limit=0.4:attack=3:release=100")

    if filters:
        filter_str = ",".join(filters)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(wav_path),
            "-af",
            filter_str,
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "0",
            str(mp3_path),
        ]
    else:
        cmd = ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-qscale:a", "0", str(mp3_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    return mp3_path


# ##################################################################
# concatenate wav files
# use ffmpeg to concatenate multiple wav files into one
# ##################################################################
# enhance output audio
# use resemble-enhance to clean up TTS output
ENHANCE_TOOL = Path.home() / "src" / "audio-enhance" / "run"


def enhance_output(input_path: Path, output_path: Path, quality: str = "ultra") -> Path:
    if not ENHANCE_TOOL.exists():
        raise RuntimeError(f"Enhancement tool not found: {ENHANCE_TOOL}")

    cmd = [str(ENHANCE_TOOL), "enhance", str(input_path), str(output_path)]
    if quality == "hq":
        cmd.append("--hq")
    elif quality == "ultra":
        cmd.append("--ultra")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Enhancement failed: {result.stderr}")

    return output_path


# ##################################################################
# memory safety threshold
# if available memory falls below this (in GB), abort before OOM crash
# set low (1.5GB) since macOS aggressively uses memory for file cache
# and will reclaim it as needed - we just want to catch truly critical situations
MEMORY_SAFETY_THRESHOLD_GB = 1.5


def get_available_memory_gb() -> float:
    """Get available system memory in GB (macOS specific).

    Includes free, speculative, and inactive (reclaimable cache) pages
    since macOS will reclaim inactive pages when apps need memory.
    """
    try:
        # use vm_stat for macOS
        result = subprocess.run(["vm_stat"], capture_output=True, text=True)
        if result.returncode != 0:
            return float("inf")  # can't check, assume OK

        lines = result.stdout.strip().split("\n")
        page_size = 4096  # default page size on macOS

        available_pages = 0
        for line in lines:
            # free pages - truly unused
            if "Pages free:" in line:
                available_pages += int(line.split(":")[1].strip().rstrip("."))
            # speculative pages - prefetched, immediately reclaimable
            elif "Pages speculative:" in line:
                available_pages += int(line.split(":")[1].strip().rstrip("."))
            # inactive pages - file cache, reclaimable when needed
            elif "Pages inactive:" in line:
                available_pages += int(line.split(":")[1].strip().rstrip("."))
            # purgeable pages - can be freed immediately
            elif "Pages purgeable:" in line:
                available_pages += int(line.split(":")[1].strip().rstrip("."))

        return (available_pages * page_size) / (1024**3)
    except Exception:
        return float("inf")  # can't check, assume OK


def check_memory_safe(operation: str = "operation") -> None:
    """Check if there's enough memory to continue. Raises if below threshold."""
    available = get_available_memory_gb()
    if available < MEMORY_SAFETY_THRESHOLD_GB:
        raise MemoryError(
            f"Aborting {operation}: only {available:.1f}GB available memory "
            f"(threshold: {MEMORY_SAFETY_THRESHOLD_GB}GB). "
            "Try closing other applications or reducing chunk count."
        )


# memory limit for subprocess restart (10GB)
SUBPROCESS_MEMORY_LIMIT_GB = 10.0
# exit code meaning "memory limit exceeded, need restart"
EXIT_CODE_MEMORY_RESTART = 77


# ##################################################################
# subprocess synthesis
# runs chunks in ONE subprocess to avoid asyncio SIGCHLD conflicts
# model loads once, gc.collect() between chunks for memory management
# if memory exceeds 10GB, exits with code 77 to signal restart needed
def _synthesize_subprocess(
    engine_type: str,
    chunks: list[str],
    output_wavs: list[str],
    language: str = "English",
    temperature: float = 0.9,
    voice: str | None = None,
    voice_description: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    model_name: str | None = None,
    start_index: int = 0,
) -> tuple[int, int]:
    """
    Internal: Synthesize chunks in ONE subprocess for memory isolation.
    Model loads once, processes chunks with gc between them.
    If memory exceeds 10GB, exits early with code 77.

    Returns (exit_code, last_completed_index).
    Exit code 0 = all done, 77 = memory restart needed, 1 = error.
    """
    import json

    chunks_json = json.dumps(chunks)
    output_wavs_json = json.dumps(output_wavs)

    script = f"""
import sys
import os
import gc
import json
import warnings
import logging
import resource

# suppress ALL warnings including asyncio event loop cleanup warnings
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

import numpy as np
import soundfile as sf
import mlx.core as mx
from mlx_audio.tts.utils import load_model

def get_memory_gb():
    \"\"\"Get current process memory usage in GB.\"\"\"
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)

MEMORY_LIMIT_GB = {SUBPROCESS_MEMORY_LIMIT_GB}
engine_type = {repr(engine_type)}
chunks = json.loads({repr(chunks_json)})
output_wavs = json.loads({repr(output_wavs_json)})
start_index = {start_index}
language = {repr(language)}
temperature = {temperature}
voice = {repr(voice)}
voice_description = {repr(voice_description)}
ref_audio = {repr(ref_audio)}
ref_text = {repr(ref_text)}
model_name = {repr(model_name)}

try:
    # load model ONCE
    if engine_type == "custom":
        model = load_model(model_name or "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")
    elif engine_type == "design":
        model = load_model(model_name or "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
    elif engine_type == "clone":
        model = load_model(model_name or "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        # load reference audio once
        ref_data, sr = sf.read(ref_audio)
        if len(ref_data.shape) > 1:
            ref_data = np.mean(ref_data, axis=1)
        if sr != model.sample_rate:
            import tempfile
            import subprocess as sp
            from pathlib import Path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                resampled_path = Path(tmp.name)
            sp.run(["ffmpeg", "-y", "-i", ref_audio, "-ar", str(model.sample_rate), "-ac", "1", str(resampled_path)],
                   capture_output=True, check=True)
            ref_data, _ = sf.read(resampled_path)
            resampled_path.unlink()
        # CRITICAL: Limit reference audio and text to ~15 seconds
        # Longer references cause the model to confuse ref_text with input text
        # and output the training script instead of the requested dialogue
        MAX_REF_SECONDS = 15
        MAX_REF_WORDS = int(MAX_REF_SECONDS * 2.5)  # ~37 words at 150 WPM

        ref_audio_duration = len(ref_data) / model.sample_rate
        if ref_audio_duration > MAX_REF_SECONDS:
            ref_data = ref_data[:int(model.sample_rate * MAX_REF_SECONDS)]
            print(f"Truncated ref_audio from {{ref_audio_duration:.1f}}s to {{MAX_REF_SECONDS}}s", flush=True)

        ref_text_words = ref_text.split()
        if len(ref_text_words) > MAX_REF_WORDS:
            ref_text = ' '.join(ref_text_words[:MAX_REF_WORDS])
            print(f"Truncated ref_text from {{len(ref_text_words)}} to {{MAX_REF_WORDS}} words", flush=True)

        ref_audio_array = mx.array(ref_data.astype(np.float32))
    else:
        raise ValueError(f"Unknown engine type: {{engine_type}}")

    # process chunks starting from start_index
    total = len(chunks)
    last_completed = start_index - 1
    for i in range(start_index, total):
        chunk_text = chunks[i]
        output_wav = output_wavs[i]

        print(f"Processing chunk {{i+1}}/{{total}}...", flush=True)

        if engine_type == "custom":
            results = list(model.generate_custom_voice(
                text=chunk_text,
                language=language,
                speaker=voice,
                temperature=temperature,
            ))
        elif engine_type == "design":
            results = list(model.generate_voice_design(
                text=chunk_text,
                language=language,
                instruct=voice_description,
                temperature=temperature,
            ))
        elif engine_type == "clone":
            results = list(model.generate(
                text=chunk_text,
                ref_audio=ref_audio_array,
                ref_text=ref_text,
                lang_code=language.lower()[:2],
                temperature=temperature,
            ))

        if not results:
            raise RuntimeError(f"No audio generated for chunk {{i+1}}")

        audio = results[0].audio
        if hasattr(audio, 'tolist'):
            audio_np = np.array(audio.tolist(), dtype=np.float32)
        else:
            audio_np = np.array(audio, dtype=np.float32)

        sf.write(output_wav, audio_np, model.sample_rate)
        last_completed = i
        print(f"Chunk {{i+1}}/{{total}} complete", flush=True)

        # cleanup between chunks
        del results, audio, audio_np
        gc.collect()
        mx.clear_cache()

        # check memory - if too high, exit for restart
        mem_gb = get_memory_gb()
        if mem_gb > MEMORY_LIMIT_GB:
            print(f"Memory limit exceeded ({{mem_gb:.1f}}GB > {{MEMORY_LIMIT_GB}}GB), restarting...", flush=True)
            # write last completed index to stdout for parent to read
            print(f"LAST_COMPLETED:{{last_completed}}", flush=True)
            del model
            gc.collect()
            mx.clear_cache()
            os._exit({EXIT_CODE_MEMORY_RESTART})

    # final cleanup
    del model
    gc.collect()
    mx.clear_cache()
    print(f"LAST_COMPLETED:{{last_completed}}", flush=True)
    os._exit(0)
except Exception as e:
    print(f"Synthesis failed: {{e}}", file=sys.stderr)
    sys.stderr.flush()
    os._exit(1)
"""

    # run in subprocess with isolated memory
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
    )

    # print stdout (progress messages)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            if not line.startswith("LAST_COMPLETED:"):
                print(line)

    # parse last completed index from output (for clean exits)
    last_completed = start_index - 1
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            if line.startswith("LAST_COMPLETED:"):
                last_completed = int(line.split(":")[1])

    # if subprocess crashed (no LAST_COMPLETED), check which files exist
    if last_completed == start_index - 1 and result.returncode != 0:
        # check which output files actually exist
        for i in range(len(output_wavs) - 1, start_index - 1, -1):
            wav_path = Path(output_wavs[i])
            if wav_path.exists() and wav_path.stat().st_size > 100:
                last_completed = i
                break

    if result.returncode != 0 and result.returncode != EXIT_CODE_MEMORY_RESTART:
        print(f"Subprocess stderr: {result.stderr}", file=sys.stderr)

    return result.returncode, last_completed


def synthesize_with_restart(
    engine_type: str,
    chunks: list[str],
    output_wavs: list[str],
    language: str = "English",
    temperature: float = 0.9,
    voice: str | None = None,
    voice_description: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    model_name: str | None = None,
) -> int:
    """
    Synthesize all chunks, restarting subprocess on any failure.
    Handles memory limit exceeded (code 77) and GPU crashes (SIGABRT, etc).
    Returns 0 on success, 1 on error.
    """
    import time

    start_index = 0
    max_restarts = 50  # safety limit
    consecutive_failures = 0
    max_consecutive_failures = 5  # give up if same chunk fails 5 times
    restart_delay_seconds = 5  # give GPU time to recover between crashes

    for restart_num in range(max_restarts):
        rc, last_completed = _synthesize_subprocess(
            engine_type=engine_type,
            chunks=chunks,
            output_wavs=output_wavs,
            language=language,
            temperature=temperature,
            voice=voice,
            voice_description=voice_description,
            ref_audio=ref_audio,
            ref_text=ref_text,
            model_name=model_name,
            start_index=start_index,
        )

        if rc == 0:
            return 0  # success

        # any failure - restart from next completed chunk
        new_start = last_completed + 1
        if new_start >= len(chunks):
            return 0  # all done despite error on last chunk

        if new_start == start_index:
            # no progress - same chunk failing repeatedly
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"Chunk {start_index + 1} failed {consecutive_failures} times, giving up", file=sys.stderr)
                return 1
            print(
                f"Chunk {start_index + 1} failed (attempt {consecutive_failures}/{max_consecutive_failures}), retrying..."
            )
        else:
            # made progress - reset failure counter
            consecutive_failures = 0
            start_index = new_start

        if rc == EXIT_CODE_MEMORY_RESTART:
            print(f"Memory limit exceeded, restarting from chunk {start_index + 1}/{len(chunks)}...")
        else:
            # GPU crash or other error (e.g., SIGABRT = -6)
            print(f"Subprocess crashed (exit {rc}), restarting from chunk {start_index + 1}/{len(chunks)}...")
            # give GPU time to recover before retrying
            if restart_num < max_restarts - 1:
                delay = restart_delay_seconds * (1 + consecutive_failures)  # exponential backoff
                print(f"Waiting {delay}s for GPU to recover...")
                time.sleep(delay)

    print(f"Max restarts ({max_restarts}) exceeded", file=sys.stderr)
    return 1


def concatenate_wav_files(wav_paths: list[Path], output_path: Path) -> Path:
    if len(wav_paths) == 1:
        # just copy the single file
        import shutil

        shutil.copy(wav_paths[0], output_path)
        return output_path

    # create concat list file for ffmpeg
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_list = Path(f.name)
        for wav_path in wav_paths:
            f.write(f"file '{wav_path}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    concat_list.unlink()

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
    return output_path


# ##################################################################
# tts engine
# abstract interface for text-to-speech engines allowing multiple model backends
class TtsEngine(ABC):
    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Path,
        language: str = "English",
        temperature: float = DEFAULT_TEMPERATURE,
        speed: float = DEFAULT_SPEED,
        enhance: bool = False,
    ) -> Path:
        # ##################################################################
        # synthesize
        # convert text to speech and write to output_path, returning the path
        # if enhance=True, apply AI enhancement to clean up the output
        ...


# ##################################################################
# qwen tts engine
# implementation using qwen3-tts models via mlx-audio for apple silicon
class QwenTtsEngine(TtsEngine):
    def __init__(
        self, model_name: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16", voice: str = DEFAULT_VOICE
    ):
        # ##################################################################
        # init
        # load the qwen tts model with specified variant and voice
        self.model_name = model_name
        if voice.lower() not in [v.lower() for v in QWEN_VOICES]:
            raise ValueError(f"Unknown voice: {voice}. Supported: {', '.join(QWEN_VOICES)}")
        self.voice = voice
        self._model = None

    def _get_model(self):
        # ##################################################################
        # get model
        # lazy load model using mlx-audio
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(self.model_name)
        return self._model

    def synthesize(
        self,
        text: str,
        output_path: Path,
        language: str = "English",
        temperature: float = DEFAULT_TEMPERATURE,
        speed: float = DEFAULT_SPEED,
        enhance: bool = False,
    ) -> Path:
        # ##################################################################
        # synthesize
        # generate speech from text using qwen model and save as mp3
        # ALL chunks processed in ONE subprocess to prevent asyncio SIGCHLD conflicts
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # check memory before starting
        check_memory_safe("custom voice synthesis")

        # split text into chunks for memory-efficient processing
        chunks = split_into_chunks(text)

        # create temp files for each chunk
        chunk_wavs: list[Path] = []
        for i in range(len(chunks)):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                chunk_wavs.append(Path(tmp.name))

        try:
            # synthesize ALL chunks in ONE subprocess
            rc = synthesize_with_restart(
                engine_type="custom",
                chunks=chunks,
                output_wavs=[str(p) for p in chunk_wavs],
                language=language,
                temperature=temperature,
                voice=self.voice,
                model_name=self.model_name,
            )

            if rc != 0:
                raise RuntimeError(f"Synthesis subprocess failed (exit code {rc})")

            # verify all chunks were created
            for i, chunk_wav in enumerate(chunk_wavs):
                if not chunk_wav.exists() or chunk_wav.stat().st_size < 100:
                    raise RuntimeError(f"No audio generated for chunk {i + 1}")

            # enhance each chunk if requested
            if enhance:
                for i, chunk_wav in enumerate(chunk_wavs):
                    print(f"Enhancing chunk {i + 1}/{len(chunks)}...")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        enhanced_chunk = Path(tmp.name)
                    enhance_output(chunk_wav, enhanced_chunk)
                    chunk_wav.unlink()
                    chunk_wavs[i] = enhanced_chunk

            # concatenate all chunk wavs into final output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                combined_wav = Path(tmp.name)
            concatenate_wav_files(chunk_wavs, combined_wav)

            # convert to mp3 if needed
            if output_path.suffix.lower() == ".mp3":
                convert_wav_to_mp3(combined_wav, output_path, normalize=True, speed=speed)
                combined_wav.unlink()
            else:
                shutil.move(str(combined_wav), str(output_path))

        finally:
            # clean up chunk temp files
            for chunk_wav in chunk_wavs:
                if chunk_wav.exists():
                    chunk_wav.unlink()

        return output_path


# ##################################################################
# voice design engine
# implementation using qwen3-tts voice design model for custom voice descriptions
class VoiceDesignEngine(TtsEngine):
    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        voice_description: str = "A clear, neutral voice.",
    ):
        # ##################################################################
        # init
        # load the voice design model with a voice description
        self.model_name = model_name
        self.voice_description = voice_description
        self._model = None

    def _get_model(self):
        # ##################################################################
        # get model
        # lazy load model using mlx-audio
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(self.model_name)
        return self._model

    def synthesize(
        self,
        text: str,
        output_path: Path,
        language: str = "English",
        temperature: float = DEFAULT_TEMPERATURE,
        speed: float = DEFAULT_SPEED,
        enhance: bool = False,
        enhance_quality: str = "ultra",
        work_dir: Path | None = None,
    ) -> Path:
        # ##################################################################
        # synthesize
        # generate speech from text using voice design model
        # each chunk processed in ISOLATED SUBPROCESS to prevent memory accumulation
        # if enhance=True, enhances each chunk after generation (sequential, not parallel)
        # if work_dir is provided, uses that directory and resumes from existing chunks
        import time

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # split text into chunks for memory-efficient processing
        chunks = split_into_chunks(text)
        total_chunks = len(chunks)

        # use provided work_dir or create new one
        if work_dir:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            resuming = True
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="tts_chunks_"))
            resuming = False

        print(f"\n{'=' * 60}")
        print(f"Processing {total_chunks} chunks (enhance: {enhance}, quality: {enhance_quality})")
        print(f"Work dir: {work_dir}")
        if resuming:
            print("Resuming from existing chunks...")
        print(f"{'=' * 60}\n")

        chunk_wavs: list[Path] = []
        gen_times: list[float] = []
        enh_times: list[float] = []
        total_start = time.time()

        try:
            # Phase 1: Generate all raw chunks (each in isolated subprocess)
            print("Phase 1: Generating raw audio chunks (subprocess isolation)")
            print("-" * 40)
            for i, chunk in enumerate(chunks):
                raw_path = work_dir / f"raw_{i:04d}.wav"
                enhanced_path = work_dir / f"enhanced_{i:04d}.wav"

                # skip if raw exists OR enhanced exists (already done)
                if raw_path.exists() and raw_path.stat().st_size > 1000:
                    print(f"[Generator] Chunk {i + 1}/{total_chunks} raw exists, skipping")
                    continue
                if enhanced_path.exists() and enhanced_path.stat().st_size > 1000:
                    print(f"[Generator] Chunk {i + 1}/{total_chunks} already enhanced, skipping")
                    continue

                # check memory before starting each chunk
                check_memory_safe(f"chunk {i + 1}/{total_chunks}")

                gen_start = time.time()
                print(f"[Generator] Starting chunk {i + 1}/{total_chunks} (subprocess)...")

                # synthesize in isolated subprocess - model loads and unloads completely
                # use single-chunk list for resumability (each chunk is independent)
                rc = synthesize_with_restart(
                    engine_type="design",
                    chunks=[chunk],
                    output_wavs=[str(raw_path)],
                    language=language,
                    temperature=temperature,
                    voice_description=self.voice_description,
                    model_name=self.model_name,
                )

                if rc != 0:
                    raise RuntimeError(f"Subprocess failed for chunk {i + 1} (exit code {rc})")

                if not raw_path.exists() or raw_path.stat().st_size < 100:
                    raise RuntimeError(f"No audio generated for chunk {i + 1}")

                # track generation time
                gen_elapsed = time.time() - gen_start
                gen_times.append(gen_elapsed)
                avg_gen = sum(gen_times) / len(gen_times)
                remaining = total_chunks - (i + 1)
                # count existing chunks we'll skip (raw or enhanced)
                existing = sum(
                    1
                    for j in range(i + 1, total_chunks)
                    if (work_dir / f"raw_{j:04d}.wav").exists() or (work_dir / f"enhanced_{j:04d}.wav").exists()
                )
                gen_eta = avg_gen * (remaining - existing)

                print(
                    f"[Generator] Completed chunk {i + 1}/{total_chunks} in {gen_elapsed:.1f}s "
                    f"(avg: {avg_gen:.1f}s, ETA: {gen_eta:.0f}s)"
                )

            # Phase 2: Enhance all chunks (if requested)
            if enhance:
                print(f"\n{'=' * 60}")
                print("Phase 2: Enhancing audio chunks")
                print("-" * 40)
                for i in range(total_chunks):
                    raw_path = work_dir / f"raw_{i:04d}.wav"
                    enhanced_path = work_dir / f"enhanced_{i:04d}.wav"

                    # skip if already enhanced
                    if enhanced_path.exists() and enhanced_path.stat().st_size > 1000:
                        print(f"[Enhancer] Chunk {i + 1}/{total_chunks} already enhanced, skipping")
                        chunk_wavs.append(enhanced_path)
                        continue

                    if not raw_path.exists():
                        raise RuntimeError(f"Raw chunk {i + 1} missing: {raw_path}")

                    enh_start = time.time()
                    print(f"[Enhancer] Starting chunk {i + 1}/{total_chunks} ({enhance_quality})...")

                    enhance_output(raw_path, enhanced_path, quality=enhance_quality)

                    enh_elapsed = time.time() - enh_start
                    enh_times.append(enh_elapsed)
                    avg_enh = sum(enh_times) / len(enh_times)
                    remaining = total_chunks - (i + 1)
                    existing = sum(
                        1 for j in range(i + 1, total_chunks) if (work_dir / f"enhanced_{j:04d}.wav").exists()
                    )
                    enh_eta = avg_enh * (remaining - existing)

                    print(
                        f"[Enhancer] Completed chunk {i + 1}/{total_chunks} in {enh_elapsed:.1f}s "
                        f"(avg: {avg_enh:.1f}s, ETA: {enh_eta:.0f}s)"
                    )

                    chunk_wavs.append(enhanced_path)
                    # delete raw after enhancing
                    raw_path.unlink(missing_ok=True)
            else:
                # no enhancement - use raw files
                for i in range(total_chunks):
                    chunk_wavs.append(work_dir / f"raw_{i:04d}.wav")

            # Phase 3: Concatenate
            print(f"\n{'=' * 60}")
            print(f"Phase 3: Concatenating {len(chunk_wavs)} chunks...")
            combined_wav = work_dir / "combined.wav"
            concatenate_wav_files(chunk_wavs, combined_wav)

            # convert to final format
            if output_path.suffix.lower() == ".mp3":
                convert_wav_to_mp3(combined_wav, output_path, normalize=True, speed=speed)
            else:
                shutil.move(str(combined_wav), str(output_path))

            # final stats
            total_elapsed = time.time() - total_start
            print(f"\n{'=' * 60}")
            print(f"Complete! Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
            if gen_times:
                print(f"Avg generation: {sum(gen_times) / len(gen_times):.1f}s/chunk ({len(gen_times)} generated)")
            if enh_times:
                print(f"Avg enhancement: {sum(enh_times) / len(enh_times):.1f}s/chunk ({len(enh_times)} enhanced)")
            print(f"Output: {output_path}")
            print(f"{'=' * 60}\n")

            # clean up work directory on success
            shutil.rmtree(work_dir, ignore_errors=True)

        except Exception as e:
            print(f"\n{'=' * 60}")
            print(f"ERROR: {e}")
            print(f"Work directory preserved for resume: {work_dir}")
            print(f"{'=' * 60}\n")
            raise

        return output_path


# ##################################################################
# voice clone engine
# implementation using qwen3-tts base model for voice cloning from audio samples
class VoiceCloneEngine(TtsEngine):
    def __init__(
        self,
        model_name: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        ref_audio: str = "",
        ref_text: str = "",
        temp_dir: Path | None = None,
    ):
        # ##################################################################
        # init
        # load the base model with reference audio and text for cloning
        # NOTE: 1.7B Base model exists but mlx-audio doesn't support it yet (missing speech_tokenizer params)
        self.model_name = model_name
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self._model = None
        self._ref_audio_array = None  # cached resampled reference audio
        self._temp_dir = temp_dir  # temp directory to clean up after synthesis

    def _get_model(self):
        # ##################################################################
        # get model
        # lazy load model using mlx-audio
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(self.model_name)
        return self._model

    def synthesize(
        self,
        text: str,
        output_path: Path,
        language: str = "English",
        temperature: float = DEFAULT_TEMPERATURE,
        speed: float = DEFAULT_SPEED,
        enhance: bool = False,
    ) -> Path:
        # ##################################################################
        # synthesize
        # generate speech from text using voice cloning
        # ALL chunks processed in ONE subprocess to prevent asyncio SIGCHLD conflicts
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # check memory before starting
        check_memory_safe("voice cloning synthesis")

        # split text into chunks for memory-efficient processing
        chunks = split_into_chunks(text)

        # create temp files for each chunk
        chunk_wavs: list[Path] = []
        for i in range(len(chunks)):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                chunk_wavs.append(Path(tmp.name))

        try:
            # synthesize ALL chunks in ONE subprocess
            rc = synthesize_with_restart(
                engine_type="clone",
                chunks=chunks,
                output_wavs=[str(p) for p in chunk_wavs],
                language=language,
                temperature=temperature,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                model_name=self.model_name,
            )

            if rc != 0:
                raise RuntimeError(f"Synthesis subprocess failed (exit code {rc})")

            # verify all chunks were created
            for i, chunk_wav in enumerate(chunk_wavs):
                if not chunk_wav.exists() or chunk_wav.stat().st_size < 100:
                    raise RuntimeError(f"No audio generated for chunk {i + 1}")

            # enhance each chunk if requested
            if enhance:
                for i, chunk_wav in enumerate(chunk_wavs):
                    print(f"Enhancing chunk {i + 1}/{len(chunks)}...")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        enhanced_chunk = Path(tmp.name)
                    enhance_output(chunk_wav, enhanced_chunk)
                    chunk_wav.unlink()
                    chunk_wavs[i] = enhanced_chunk

            # concatenate all chunk wavs into final output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                combined_wav = Path(tmp.name)
            concatenate_wav_files(chunk_wavs, combined_wav)

            # convert to mp3 if needed
            if output_path.suffix.lower() == ".mp3":
                convert_wav_to_mp3(combined_wav, output_path, normalize=True, speed=speed)
                combined_wav.unlink()
            else:
                shutil.move(str(combined_wav), str(output_path))

        finally:
            # clean up chunk temp files
            for chunk_wav in chunk_wavs:
                if chunk_wav.exists():
                    chunk_wav.unlink()
            # clean up temp directory from zip extraction
            if self._temp_dir and self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)

        return output_path


# ##################################################################
# load engine from zip
# extract voice package zip and return VoiceCloneEngine
def load_engine_from_zip(zip_path: str) -> "VoiceCloneEngine":
    import zipfile

    zip_file = Path(zip_path)
    if not zip_file.exists():
        raise FileNotFoundError(f"Voice package not found: {zip_path}")

    # extract to temp directory
    extract_dir = Path(tempfile.mkdtemp(prefix="voice_pkg_"))

    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(extract_dir)

    voice_wav = extract_dir / "voice.wav"
    voice_txt = extract_dir / "voice.txt"

    if not voice_wav.exists():
        raise ValueError(f"Voice package missing voice.wav: {zip_path}")
    if not voice_txt.exists():
        raise ValueError(f"Voice package missing voice.txt: {zip_path}")

    ref_text = voice_txt.read_text().strip()

    return VoiceCloneEngine(
        ref_audio=str(voice_wav),
        ref_text=ref_text,
        temp_dir=extract_dir,
    )


# ##################################################################
# get engine
# factory function to get a tts engine by model name and voice
def get_engine(model: str = "qwen", voice: str = DEFAULT_VOICE, voice_description: str | None = None) -> TtsEngine:
    # ##################################################################
    # get engine
    # return the appropriate engine implementation for the given model and voice
    # if voice_description is provided, use VoiceDesignEngine
    # if voice is a registered custom voice name, look up its description or ref_audio

    # check if voice is a zip file (portable voice package)
    if voice.endswith(".zip"):
        return load_engine_from_zip(voice)

    # check if voice is a registered custom voice
    custom_voice = get_voice_description(voice)
    if custom_voice:
        voice_type = custom_voice.get("type", "description")
        if voice_type == "clone":
            # voice cloning from audio samples (requires ref_audio and ref_text)
            return VoiceCloneEngine(
                ref_audio=custom_voice["ref_audio"],
                ref_text=custom_voice.get("ref_text", ""),
            )
        else:
            # voice design from description
            return VoiceDesignEngine(voice_description=custom_voice["description"])

    # explicit voice description provided
    if voice_description:
        return VoiceDesignEngine(voice_description=voice_description)

    # standard qwen voices
    if model == "qwen" or model.startswith("mlx-community/"):
        model_name = (
            model if model.startswith("mlx-community/") else "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
        )
        return QwenTtsEngine(model_name=model_name, voice=voice)

    raise ValueError(f"Unknown model: {model}. Supported: qwen")


# ##################################################################
# voice validation
# check if a voice name is valid (built-in or custom)
def is_valid_voice(voice: str) -> bool:
    """Check if a voice name is valid (built-in or registered custom voice)."""
    # built-in voices
    if voice.lower() in [v.lower() for v in QWEN_VOICES]:
        return True
    # custom registered voices
    if get_voice_description(voice) is not None:
        return True
    # portable voice package
    if voice.endswith(".zip") and Path(voice).exists():
        return True
    return False


def get_all_valid_voices() -> list[str]:
    """Return list of all valid voice names (built-in + custom)."""
    voices = list(QWEN_VOICES)
    voices.extend(list_custom_voices())
    return voices


# ##################################################################
# multi-speaker TTS
# parse JSONL input and synthesize with multiple voices
def parse_multi_speaker_jsonl(jsonl_path: Path) -> list[tuple[str, str]]:
    """
    Parse a JSONL file with multi-speaker dialogue.
    Each line should be a JSON object with exactly one key (voice name) and value (text).
    Example: {"bob": "Hello there"}\n{"jane": "Well hello to you"}

    Returns list of (voice, text) tuples.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    result: list[tuple[str, str]] = []

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue  # skip empty lines

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {line_num}: {e}")

        if not isinstance(obj, dict):
            raise ValueError(f"Line {line_num} must be a JSON object, got {type(obj).__name__}")

        if len(obj) != 1:
            raise ValueError(f"Line {line_num} must have exactly one key (voice name), got {len(obj)}")

        voice = list(obj.keys())[0]
        text = obj[voice]

        if not isinstance(text, str):
            raise ValueError(f"Line {line_num}: text must be a string, got {type(text).__name__}")

        if text.strip():
            result.append((voice, text.strip()))

    return result


def validate_multi_speaker_voices(segments: list[tuple[str, str]]) -> None:
    """
    Validate that all voices in segments are valid.
    Raises ValueError with clear message listing all invalid voices if any are found.
    """
    invalid_voices: set[str] = set()
    for voice, _ in segments:
        if not is_valid_voice(voice):
            invalid_voices.add(voice)

    if invalid_voices:
        valid_voices = get_all_valid_voices()
        raise ValueError(
            f"Unknown voice(s): {', '.join(sorted(invalid_voices))}. Valid voices: {', '.join(sorted(valid_voices))}"
        )


def group_consecutive_speakers(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Group consecutive segments with the same speaker.
    Multiple consecutive {"bob": "text1"}, {"bob": "text2"} become one ("bob", "text1 text2").

    Returns list of (voice, combined_text) tuples.
    """
    if not segments:
        return []

    grouped: list[tuple[str, str]] = []
    current_voice, current_text = segments[0]

    for voice, text in segments[1:]:
        if voice.lower() == current_voice.lower():
            # same speaker - append text
            current_text = current_text + " " + text
        else:
            # different speaker - save current and start new
            grouped.append((current_voice, current_text))
            current_voice = voice
            current_text = text

    # don't forget the last group
    grouped.append((current_voice, current_text))

    return grouped


def synthesize_multi_speaker(
    jsonl_path: Path,
    output_path: Path,
    language: str = "English",
    temperature: float = DEFAULT_TEMPERATURE,
    speed: float = DEFAULT_SPEED,
    enhance: bool = False,
) -> Path:
    """
    Synthesize multi-speaker dialogue from a JSONL file.

    Input format: JSONL with lines like {"bob": "Hello"}\n{"jane": "Hi there"}
    Each line's key is the voice name, value is the text to speak.

    Validates ALL voices upfront before starting synthesis.
    Groups consecutive same-speaker lines for efficiency.
    Generates each speaker segment separately and concatenates.

    Args:
        jsonl_path: Path to JSONL input file
        output_path: Path for output audio file (wav or mp3)
        language: Language for synthesis
        temperature: Synthesis temperature
        speed: Speed multiplier (applied at end)
        enhance: Whether to enhance audio quality

    Returns:
        Path to output audio file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # parse input
    segments = parse_multi_speaker_jsonl(jsonl_path)
    if not segments:
        raise ValueError(f"No dialogue found in {jsonl_path}")

    # validate ALL voices upfront (fail fast)
    validate_multi_speaker_voices(segments)

    # group consecutive same-speaker lines
    grouped = group_consecutive_speakers(segments)

    print(f"Multi-speaker synthesis: {len(grouped)} speaker segments")
    for i, (voice, text) in enumerate(grouped):
        word_count = len(text.split())
        print(f"  [{i + 1}] {voice}: {word_count} words")

    # check memory before starting
    check_memory_safe("multi-speaker synthesis")

    # synthesize each segment
    segment_wavs: list[Path] = []

    try:
        for i, (voice, text) in enumerate(grouped):
            print(f"\nProcessing segment {i + 1}/{len(grouped)}: {voice}")

            # create temp file for this segment
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                segment_wav = Path(tmp.name)
            segment_wavs.append(segment_wav)

            # get engine for this voice
            engine = get_engine(voice=voice)

            # synthesize (engine handles chunking internally)
            engine.synthesize(
                text=text,
                output_path=segment_wav,
                language=language,
                temperature=temperature,
                speed=1.0,  # apply speed at the end for consistency
                enhance=enhance,
            )

            print(f"Segment {i + 1}/{len(grouped)} complete")

        # concatenate all segments
        print(f"\nConcatenating {len(segment_wavs)} segments...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            combined_wav = Path(tmp.name)
        concatenate_wav_files(segment_wavs, combined_wav)

        # convert to final format
        if output_path.suffix.lower() == ".mp3":
            convert_wav_to_mp3(combined_wav, output_path, normalize=True, speed=speed)
            combined_wav.unlink()
        else:
            if speed != 1.0:
                # for wav output with speed adjustment, need to process
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    speed_adjusted = Path(tmp.name)
                # use ffmpeg atempo for speed
                cmd = ["ffmpeg", "-y", "-i", str(combined_wav), "-af", f"atempo={speed}", str(speed_adjusted)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Speed adjustment failed: {result.stderr}")
                shutil.move(str(speed_adjusted), str(output_path))
                combined_wav.unlink()
            else:
                shutil.move(str(combined_wav), str(output_path))

        print(f"\nMulti-speaker synthesis complete: {output_path}")
        return output_path

    finally:
        # clean up segment temp files
        for wav in segment_wavs:
            if wav.exists():
                wav.unlink()


# ##################################################################
# tts engine module
# provides text-to-speech synthesis with pluggable model backends,
# using mlx-audio for efficient inference on apple silicon
