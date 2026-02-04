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
        # CRITICAL: Limit reference audio to ~10 seconds and DO NOT pass ref_text
        # Passing ref_text causes the model to confuse it with input text and
        # output the training script instead of the requested dialogue
        MAX_REF_SECONDS = 10

        ref_audio_duration = len(ref_data) / model.sample_rate
        if ref_audio_duration > MAX_REF_SECONDS:
            ref_data = ref_data[:int(model.sample_rate * MAX_REF_SECONDS)]
            print(f"Truncated ref_audio from {{ref_audio_duration:.1f}}s to {{MAX_REF_SECONDS}}s", flush=True)

        # Clear ref_text - passing it causes content leakage into output
        ref_text = None

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

        # calculate max_tokens based on text length
        # ~2.5 words/second at 12Hz = ~5 tokens/word, with 2x buffer
        word_count = len(chunk_text.split())
        max_tokens = max(200, word_count * 10)  # min 200, ~10 tokens per word

        if engine_type == "custom":
            results = list(model.generate_custom_voice(
                text=chunk_text,
                language=language,
                speaker=voice,
                temperature=temperature,
                max_tokens=max_tokens,
            ))
        elif engine_type == "design":
            results = list(model.generate_voice_design(
                text=chunk_text,
                language=language,
                instruct=voice_description,
                temperature=temperature,
                max_tokens=max_tokens,
            ))
        elif engine_type == "clone":
            results = list(model.generate(
                text=chunk_text,
                ref_audio=ref_audio_array,
                ref_text=ref_text,
                lang_code=language.lower()[:2],
                temperature=temperature,
                max_tokens=max_tokens,
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


def trim_silence(wav_path: Path, output_path: Path | None = None, threshold_db: float = -50, min_silence_duration: float = 0.1) -> Path:
    """
    Trim leading and trailing silence from a WAV file using ffmpeg silenceremove filter.

    Args:
        wav_path: Input WAV file
        output_path: Output path (defaults to overwriting input)
        threshold_db: Volume threshold below which is considered silence (default -50dB)
        min_silence_duration: Minimum duration of silence to remove (default 0.1s)

    Returns:
        Path to trimmed file
    """
    if output_path is None:
        output_path = wav_path

    # use temp file if overwriting
    if output_path == wav_path:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = Path(tmp.name)
    else:
        temp_path = output_path

    # silenceremove filter: remove both leading and trailing silence
    # start_periods=1 removes leading silence (stop after 1 period of non-silence)
    # stop_periods=-1 removes all trailing silence
    filter_str = (
        f"silenceremove="
        f"start_periods=1:start_duration={min_silence_duration}:start_threshold={threshold_db}dB:"
        f"stop_periods=-1:stop_duration={min_silence_duration}:stop_threshold={threshold_db}dB"
    )

    cmd = ["ffmpeg", "-y", "-i", str(wav_path), "-af", filter_str, str(temp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # if trimming fails, just keep original
        if temp_path != output_path:
            temp_path.unlink(missing_ok=True)
        return wav_path

    # if we used a temp file, move it to the original
    if output_path == wav_path:
        shutil.move(str(temp_path), str(output_path))

    return output_path


def concatenate_wav_files(wav_paths: list[Path], output_path: Path, trim_trailing_silence: bool = False) -> Path:
    """
    Concatenate multiple WAV files into one.

    Args:
        wav_paths: List of WAV files to concatenate
        output_path: Output file path
        trim_trailing_silence: If True, trim trailing silence from each file before concatenating

    Returns:
        Path to output file
    """
    if len(wav_paths) == 1:
        # just copy the single file
        if trim_trailing_silence:
            shutil.copy(wav_paths[0], output_path)
            trim_silence(output_path)
        else:
            shutil.copy(wav_paths[0], output_path)
        return output_path

    # optionally trim silence from each file first
    if trim_trailing_silence:
        for wav_path in wav_paths:
            trim_silence(wav_path)

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
    work_dir: Path | None = None,
) -> Path:
    """
    Synthesize multi-speaker dialogue from a JSONL file.

    Input format: JSONL with lines like {"bob": "Hello"}\n{"jane": "Hi there"}
    Each line's key is the voice name, value is the text to speak.

    OPTIMIZATION: Batches all lines by speaker to minimize model loads.
    Instead of loading/unloading voices for each dialogue turn, we:
    1. Group ALL lines by speaker (not just consecutive ones)
    2. For each speaker, load the model ONCE and generate ALL their lines
    3. Reassemble in original dialogue order at the end

    This reduces model loads from O(n) to O(speakers), typically 143 → 4.

    RESUMABILITY: If work_dir is provided, saves progress there and can resume.
    Each line is saved as line_{index:04d}.wav. On restart, existing files are skipped.

    Args:
        jsonl_path: Path to JSONL input file
        output_path: Path for output audio file (wav or mp3)
        language: Language for synthesis
        temperature: Synthesis temperature
        speed: Speed multiplier (applied at end)
        enhance: Whether to enhance audio quality
        work_dir: Optional directory for progress tracking (enables resume)

    Returns:
        Path to output audio file
    """
    import time

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # parse input - get (voice, text) tuples with their original indices
    segments = parse_multi_speaker_jsonl(jsonl_path)
    if not segments:
        raise ValueError(f"No dialogue found in {jsonl_path}")

    # validate ALL voices upfront (fail fast)
    validate_multi_speaker_voices(segments)

    # create indexed segments: [(index, voice, text), ...]
    indexed_segments = [(i, voice, text) for i, (voice, text) in enumerate(segments)]
    total_lines = len(indexed_segments)

    # group by speaker: {voice: [(index, text), ...]}
    speaker_batches: dict[str, list[tuple[int, str]]] = {}
    for idx, voice, text in indexed_segments:
        voice_lower = voice.lower()
        if voice_lower not in speaker_batches:
            speaker_batches[voice_lower] = []
        speaker_batches[voice_lower].append((idx, text))

    print(f"\n{'=' * 60}")
    print(f"Multi-speaker synthesis: {total_lines} lines, {len(speaker_batches)} speakers")
    print("Speaker batching enabled (optimized: one model load per speaker)")
    for voice, lines in speaker_batches.items():
        word_count = sum(len(text.split()) for _, text in lines)
        print(f"  {voice}: {len(lines)} lines, {word_count} words")
    print(f"{'=' * 60}\n")

    # set up work directory for progress tracking
    if work_dir:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        resuming = True
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="multi_speaker_"))
        resuming = False

    if resuming:
        print(f"Work directory: {work_dir}")
        existing = sum(1 for i in range(total_lines) if (work_dir / f"line_{i:04d}.wav").exists())
        if existing > 0:
            print(f"Resuming: {existing}/{total_lines} lines already complete")

    # check memory before starting
    check_memory_safe("multi-speaker synthesis")

    total_start = time.time()

    try:
        # process each speaker batch (one model load per speaker)
        for speaker_num, (voice, lines) in enumerate(speaker_batches.items(), 1):
            # check which lines for this speaker still need generation
            pending_lines = [(idx, text) for idx, text in lines
                            if not (work_dir / f"line_{idx:04d}.wav").exists()]

            if not pending_lines:
                print(f"\n[Speaker {speaker_num}/{len(speaker_batches)}] {voice}: all {len(lines)} lines complete, skipping")
                continue

            print(f"\n[Speaker {speaker_num}/{len(speaker_batches)}] {voice}: generating {len(pending_lines)}/{len(lines)} lines")

            # get voice config
            voice_config = get_voice_description(voice)
            if not voice_config:
                raise ValueError(f"Voice config not found: {voice}")

            # split long lines into chunks (<600 words each for efficiency)
            # track: [(line_idx, chunk_idx, text, output_path), ...]
            all_chunks: list[tuple[int, int, str, str]] = []
            for idx, text in pending_lines:
                chunks = split_into_chunks(text)
                if len(chunks) == 1:
                    # single chunk - output directly to line wav
                    all_chunks.append((idx, 0, chunks[0], str(work_dir / f"line_{idx:04d}.wav")))
                else:
                    # multiple chunks - use chunk wavs, concatenate later
                    for chunk_idx, chunk_text in enumerate(chunks):
                        chunk_path = str(work_dir / f"line_{idx:04d}_chunk_{chunk_idx:02d}.wav")
                        all_chunks.append((idx, chunk_idx, chunk_text, chunk_path))

            # prepare synthesis inputs
            texts = [text for _, _, text, _ in all_chunks]
            output_wavs = [path for _, _, _, path in all_chunks]

            # synthesize all chunks for this speaker in one subprocess batch
            # the subprocess loads the model ONCE and processes all chunks
            speaker_start = time.time()

            rc = synthesize_with_restart(
                engine_type="clone",
                chunks=texts,
                output_wavs=output_wavs,
                language=language,
                temperature=temperature,
                ref_audio=voice_config.get("ref_audio", ""),
                ref_text=voice_config.get("ref_text", ""),
                model_name=voice_config.get("model_name"),
            )

            if rc != 0:
                # check how many actually completed
                completed = sum(1 for path in output_wavs if Path(path).exists())
                print(f"Warning: synthesis returned {rc}, {completed}/{len(all_chunks)} chunks completed")
                if completed == 0:
                    raise RuntimeError(f"Synthesis failed for speaker {voice} (exit code {rc})")

            # concatenate multi-chunk lines into single line wavs
            lines_with_chunks: dict[int, list[tuple[int, str]]] = {}
            for line_idx, chunk_idx, _, path in all_chunks:
                if line_idx not in lines_with_chunks:
                    lines_with_chunks[line_idx] = []
                lines_with_chunks[line_idx].append((chunk_idx, path))

            for line_idx, chunk_info in lines_with_chunks.items():
                if len(chunk_info) > 1:
                    # sort by chunk index and concatenate
                    chunk_info.sort(key=lambda x: x[0])
                    chunk_paths = [Path(p) for _, p in chunk_info]
                    # verify all chunks exist
                    if all(p.exists() for p in chunk_paths):
                        line_wav = work_dir / f"line_{line_idx:04d}.wav"
                        concatenate_wav_files(chunk_paths, line_wav)
                        # clean up chunk files
                        for p in chunk_paths:
                            p.unlink()

            speaker_elapsed = time.time() - speaker_start
            completed_count = sum(1 for idx, _ in pending_lines if (work_dir / f"line_{idx:04d}.wav").exists())
            print(f"[Speaker {speaker_num}/{len(speaker_batches)}] {voice}: {completed_count} lines in {speaker_elapsed:.1f}s")

        # verify all lines exist
        missing = [i for i in range(total_lines) if not (work_dir / f"line_{i:04d}.wav").exists()]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} lines after synthesis: {missing[:10]}...")

        # concatenate in original dialogue order
        # trim trailing silence from each line for tight dialogue
        print(f"\nTrimming silence and concatenating {total_lines} lines in dialogue order...")
        line_wavs = [work_dir / f"line_{i:04d}.wav" for i in range(total_lines)]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            combined_wav = Path(tmp.name)
        concatenate_wav_files(line_wavs, combined_wav, trim_trailing_silence=True)

        # enhance if requested (on combined audio)
        if enhance:
            print("Enhancing combined audio...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                enhanced_wav = Path(tmp.name)
            enhance_output(combined_wav, enhanced_wav)
            combined_wav.unlink()
            combined_wav = enhanced_wav

        # convert to final format
        if output_path.suffix.lower() == ".mp3":
            convert_wav_to_mp3(combined_wav, output_path, normalize=True, speed=speed)
            combined_wav.unlink()
        else:
            if speed != 1.0:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    speed_adjusted = Path(tmp.name)
                cmd = ["ffmpeg", "-y", "-i", str(combined_wav), "-af", f"atempo={speed}", str(speed_adjusted)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Speed adjustment failed: {result.stderr}")
                shutil.move(str(speed_adjusted), str(output_path))
                combined_wav.unlink()
            else:
                shutil.move(str(combined_wav), str(output_path))

        total_elapsed = time.time() - total_start
        print(f"\n{'=' * 60}")
        print("Multi-speaker synthesis complete!")
        print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
        print(f"  Lines: {total_lines}")
        print(f"  Speakers: {len(speaker_batches)}")
        print(f"  Output: {output_path}")
        print(f"{'=' * 60}\n")

        # clean up work directory on success (unless user provided it)
        if not resuming:
            shutil.rmtree(work_dir, ignore_errors=True)

        return output_path

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"ERROR: {e}")
        print(f"Work directory preserved for resume: {work_dir}")
        print(f"{'=' * 60}\n")
        raise


# ##################################################################
# tts engine module
# provides text-to-speech synthesis with pluggable model backends,
# using mlx-audio for efficient inference on apple silicon
