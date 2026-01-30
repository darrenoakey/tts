import gc
import json
import logging
import multiprocessing
import re
import subprocess
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import soundfile as sf

# suppress transformers warnings about model type and tokenizer regex
# must be configured before transformers is imported by mlx_audio
logging.getLogger('transformers').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='transformers')


# available voices for qwen3-tts customvoice model (mlx-audio)
# english: aiden, ryan, ono_anna, sohee
# chinese: vivian, serena, uncle_fu, dylan (beijing dialect), eric (sichuan dialect)
QWEN_VOICES = ["aiden", "ryan", "ono_anna", "sohee", "vivian", "serena", "uncle_fu", "dylan", "eric"]
DEFAULT_VOICE = "aiden"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_SPEED = 1.0

# chunk size for processing long texts (in sentences)
# smaller chunks = more stable memory but slightly more overhead
DEFAULT_CHUNK_SENTENCES = 3

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
# splits text into chunks of approximately n sentences each
def split_into_chunks(text: str, sentences_per_chunk: int = DEFAULT_CHUNK_SENTENCES) -> list[str]:
    text = text.strip()
    if not text:
        return []

    # split on sentence boundaries (., !, ?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= sentences_per_chunk:
        return [text]

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)
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
            "ffmpeg", "-y", "-i", str(wav_path),
            "-af", filter_str,
            "-codec:a", "libmp3lame", "-qscale:a", "0",
            str(mp3_path)
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-qscale:a", "0",
            str(mp3_path)
        ]
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
# enhancer worker process
# runs in separate process, enhances chunks as they become available
def _enhancer_worker(queue: multiprocessing.Queue, status: dict, quality: str = "ultra") -> None:
    import time
    while True:
        item = queue.get()
        if item is None:
            # sentinel - done
            break
        chunk_idx, total_chunks, raw_path, enhanced_path = item
        start_time = time.time()
        status["current"] = chunk_idx
        status["state"] = "enhancing"
        print(f"[Enhancer] Starting chunk {chunk_idx}/{total_chunks}...")
        try:
            enhance_output(Path(raw_path), Path(enhanced_path), quality)
            elapsed = time.time() - start_time
            status["completed"] = chunk_idx
            status["times"].append(elapsed)
            avg_time = sum(status["times"]) / len(status["times"])
            remaining = total_chunks - chunk_idx
            eta = avg_time * remaining
            print(f"[Enhancer] Completed chunk {chunk_idx}/{total_chunks} in {elapsed:.1f}s (avg: {avg_time:.1f}s, ETA: {eta:.0f}s)")
            # delete raw file after enhancement
            Path(raw_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"[Enhancer] Failed chunk {chunk_idx}: {e}")
    status["state"] = "done"


def concatenate_wav_files(wav_paths: list[Path], output_path: Path) -> Path:
    if len(wav_paths) == 1:
        # just copy the single file
        import shutil
        shutil.copy(wav_paths[0], output_path)
        return output_path

    # create concat list file for ffmpeg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_list = Path(f.name)
        for wav_path in wav_paths:
            f.write(f"file '{wav_path}'\n")

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
    return output_path


# ##################################################################
# tts engine
# abstract interface for text-to-speech engines allowing multiple model backends
class TtsEngine(ABC):

    @abstractmethod
    def synthesize(self, text: str, output_path: Path, language: str = "English",
                   temperature: float = DEFAULT_TEMPERATURE, speed: float = DEFAULT_SPEED,
                   enhance: bool = False) -> Path:
        # ##################################################################
        # synthesize
        # convert text to speech and write to output_path, returning the path
        # if enhance=True, apply AI enhancement to clean up the output
        ...


# ##################################################################
# qwen tts engine
# implementation using qwen3-tts models via mlx-audio for apple silicon
class QwenTtsEngine(TtsEngine):

    def __init__(self, model_name: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16", voice: str = DEFAULT_VOICE):
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

    def synthesize(self, text: str, output_path: Path, language: str = "English",
                   temperature: float = DEFAULT_TEMPERATURE, speed: float = DEFAULT_SPEED,
                   enhance: bool = False) -> Path:
        # ##################################################################
        # synthesize
        # generate speech from text using qwen model and save as mp3
        # processes text in chunks to avoid memory spikes on long inputs
        # if enhance=True, apply AI enhancement to clean up the output
        import mlx.core as mx

        model = self._get_model()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # split text into chunks for memory-efficient processing
        chunks = split_into_chunks(text)
        chunk_wavs: list[Path] = []

        try:
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i + 1}/{len(chunks)}...")

                # generate audio using mlx-audio custom voice API
                results = list(model.generate_custom_voice(
                    text=chunk,
                    language=language,
                    speaker=self.voice,
                    temperature=temperature,
                ))

                if not results:
                    raise RuntimeError(f"No audio generated for chunk {i + 1}")

                # convert mlx array to numpy for soundfile
                audio = results[0].audio
                if hasattr(audio, 'tolist'):
                    # mlx array - convert to numpy
                    audio_np = np.array(audio.tolist(), dtype=np.float32)
                else:
                    audio_np = np.array(audio, dtype=np.float32)

                # write chunk to temp file immediately to free memory
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    chunk_wav = Path(tmp.name)
                sf.write(str(chunk_wav), audio_np, model.sample_rate)

                # enhance each chunk individually (faster than enhancing combined file)
                if enhance:
                    print(f"Enhancing chunk {i + 1}/{len(chunks)}...")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        enhanced_chunk = Path(tmp.name)
                    enhance_output(chunk_wav, enhanced_chunk)
                    chunk_wav.unlink()
                    chunk_wav = enhanced_chunk

                chunk_wavs.append(chunk_wav)

                # free memory from this chunk
                del results, audio, audio_np
                gc.collect()
                mx.clear_cache()

            # concatenate all chunk wavs into final output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                combined_wav = Path(tmp.name)
            concatenate_wav_files(chunk_wavs, combined_wav)

            # convert to mp3 if needed
            if output_path.suffix.lower() == ".mp3":
                convert_wav_to_mp3(combined_wav, output_path, normalize=True, speed=speed)
                combined_wav.unlink()
            else:
                import shutil
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

    def __init__(self, model_name: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
                 voice_description: str = "A clear, neutral voice."):
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

    def synthesize(self, text: str, output_path: Path, language: str = "English",
                   temperature: float = DEFAULT_TEMPERATURE, speed: float = DEFAULT_SPEED,
                   enhance: bool = False, work_dir: Path | None = None) -> Path:
        # ##################################################################
        # synthesize
        # generate speech from text using voice design model
        # if enhance=True, enhances each chunk after generation (sequential, not parallel)
        # if work_dir is provided, uses that directory and resumes from existing chunks
        import time
        import mlx.core as mx

        model = self._get_model()
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

        print(f"\n{'='*60}")
        print(f"Processing {total_chunks} chunks (enhance: {enhance})")
        print(f"Work dir: {work_dir}")
        if resuming:
            print("Resuming from existing chunks...")
        print(f"{'='*60}\n")

        chunk_wavs: list[Path] = []
        gen_times: list[float] = []
        enh_times: list[float] = []
        total_start = time.time()

        try:
            # Phase 1: Generate all raw chunks
            print("Phase 1: Generating raw audio chunks")
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

                gen_start = time.time()
                print(f"[Generator] Starting chunk {i + 1}/{total_chunks}...")

                # generate audio using voice design API
                results = list(model.generate_voice_design(
                    text=chunk,
                    language=language,
                    instruct=self.voice_description,
                    temperature=temperature,
                ))

                if not results:
                    raise RuntimeError(f"No audio generated for chunk {i + 1}")

                # convert mlx array to numpy for soundfile
                audio = results[0].audio
                if hasattr(audio, 'tolist'):
                    audio_np = np.array(audio.tolist(), dtype=np.float32)
                else:
                    audio_np = np.array(audio, dtype=np.float32)

                sf.write(str(raw_path), audio_np, model.sample_rate)

                # track generation time
                gen_elapsed = time.time() - gen_start
                gen_times.append(gen_elapsed)
                avg_gen = sum(gen_times) / len(gen_times)
                remaining = total_chunks - (i + 1)
                # count existing chunks we'll skip (raw or enhanced)
                existing = sum(1 for j in range(i + 1, total_chunks)
                              if (work_dir / f"raw_{j:04d}.wav").exists()
                              or (work_dir / f"enhanced_{j:04d}.wav").exists())
                gen_eta = avg_gen * (remaining - existing)

                print(f"[Generator] Completed chunk {i + 1}/{total_chunks} in {gen_elapsed:.1f}s "
                      f"(avg: {avg_gen:.1f}s, ETA: {gen_eta:.0f}s)")

                # free memory
                del results, audio, audio_np
                gc.collect()
                mx.clear_cache()

            # Phase 2: Enhance all chunks (if requested)
            if enhance:
                print(f"\n{'='*60}")
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
                    print(f"[Enhancer] Starting chunk {i + 1}/{total_chunks}...")

                    enhance_output(raw_path, enhanced_path)

                    enh_elapsed = time.time() - enh_start
                    enh_times.append(enh_elapsed)
                    avg_enh = sum(enh_times) / len(enh_times)
                    remaining = total_chunks - (i + 1)
                    existing = sum(1 for j in range(i + 1, total_chunks)
                                  if (work_dir / f"enhanced_{j:04d}.wav").exists())
                    enh_eta = avg_enh * (remaining - existing)

                    print(f"[Enhancer] Completed chunk {i + 1}/{total_chunks} in {enh_elapsed:.1f}s "
                          f"(avg: {avg_enh:.1f}s, ETA: {enh_eta:.0f}s)")

                    chunk_wavs.append(enhanced_path)
                    # delete raw after enhancing
                    raw_path.unlink(missing_ok=True)
            else:
                # no enhancement - use raw files
                for i in range(total_chunks):
                    chunk_wavs.append(work_dir / f"raw_{i:04d}.wav")

            # Phase 3: Concatenate
            print(f"\n{'='*60}")
            print(f"Phase 3: Concatenating {len(chunk_wavs)} chunks...")
            combined_wav = work_dir / "combined.wav"
            concatenate_wav_files(chunk_wavs, combined_wav)

            # convert to final format
            if output_path.suffix.lower() == ".mp3":
                convert_wav_to_mp3(combined_wav, output_path, normalize=True, speed=speed)
            else:
                import shutil
                shutil.move(str(combined_wav), str(output_path))

            # final stats
            total_elapsed = time.time() - total_start
            print(f"\n{'='*60}")
            print(f"Complete! Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
            if gen_times:
                print(f"Avg generation: {sum(gen_times)/len(gen_times):.1f}s/chunk ({len(gen_times)} generated)")
            if enh_times:
                print(f"Avg enhancement: {sum(enh_times)/len(enh_times):.1f}s/chunk ({len(enh_times)} enhanced)")
            print(f"Output: {output_path}")
            print(f"{'='*60}\n")

            # clean up work directory on success
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR: {e}")
            print(f"Work directory preserved for resume: {work_dir}")
            print(f"{'='*60}\n")
            raise

        return output_path


# ##################################################################
# voice clone engine
# implementation using qwen3-tts base model for voice cloning from audio samples
class VoiceCloneEngine(TtsEngine):

    def __init__(self, model_name: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                 ref_audio: str = "", ref_text: str = ""):
        # ##################################################################
        # init
        # load the base model with reference audio and text for cloning
        # NOTE: 1.7B Base model exists but mlx-audio doesn't support it yet (missing speech_tokenizer params)
        self.model_name = model_name
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self._model = None
        self._ref_audio_array = None  # cached resampled reference audio

    def _get_model(self):
        # ##################################################################
        # get model
        # lazy load model using mlx-audio
        if self._model is None:
            from mlx_audio.tts.utils import load_model
            self._model = load_model(self.model_name)
        return self._model

    def synthesize(self, text: str, output_path: Path, language: str = "English",
                   temperature: float = DEFAULT_TEMPERATURE, speed: float = DEFAULT_SPEED,
                   enhance: bool = False) -> Path:
        # ##################################################################
        # synthesize
        # generate speech from text using voice cloning
        import mlx.core as mx

        model = self._get_model()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # load and cache reference audio as mlx array (resampled to model sample rate)
        if self._ref_audio_array is None:
            ref_data, sr = sf.read(self.ref_audio)
            # convert stereo to mono if needed
            if len(ref_data.shape) > 1:
                ref_data = np.mean(ref_data, axis=1)
            # resample to model sample rate (24kHz) if needed
            if sr != model.sample_rate:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    resampled_path = Path(tmp.name)
                cmd = ["ffmpeg", "-y", "-i", self.ref_audio, "-ar", str(model.sample_rate), "-ac", "1", str(resampled_path)]
                subprocess.run(cmd, capture_output=True, check=True)
                ref_data, _ = sf.read(resampled_path)
                resampled_path.unlink()
            self._ref_audio_array = mx.array(ref_data.astype(np.float32))
        ref_audio_array = self._ref_audio_array

        # split text into chunks for memory-efficient processing
        chunks = split_into_chunks(text)
        chunk_wavs: list[Path] = []

        try:
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i + 1}/{len(chunks)}...")

                # generate audio using voice cloning API
                # ref_text is required for voice cloning to work
                results = list(model.generate(
                    text=chunk,
                    ref_audio=ref_audio_array,
                    ref_text=self.ref_text,
                    lang_code=language.lower()[:2],  # "en" for English
                    temperature=temperature,
                ))

                if not results:
                    raise RuntimeError(f"No audio generated for chunk {i + 1}")

                # convert mlx array to numpy for soundfile
                audio = results[0].audio
                if hasattr(audio, 'tolist'):
                    audio_np = np.array(audio.tolist(), dtype=np.float32)
                else:
                    audio_np = np.array(audio, dtype=np.float32)

                # write chunk to temp file immediately to free memory
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    chunk_wav = Path(tmp.name)
                sf.write(str(chunk_wav), audio_np, model.sample_rate)

                # enhance each chunk individually (faster than enhancing combined file)
                if enhance:
                    print(f"Enhancing chunk {i + 1}/{len(chunks)}...")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        enhanced_chunk = Path(tmp.name)
                    enhance_output(chunk_wav, enhanced_chunk)
                    chunk_wav.unlink()
                    chunk_wav = enhanced_chunk

                chunk_wavs.append(chunk_wav)

                # free memory from this chunk
                del results, audio, audio_np
                gc.collect()
                mx.clear_cache()

            # concatenate all chunk wavs into final output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                combined_wav = Path(tmp.name)
            concatenate_wav_files(chunk_wavs, combined_wav)

            # convert to mp3 if needed
            if output_path.suffix.lower() == ".mp3":
                convert_wav_to_mp3(combined_wav, output_path, normalize=True, speed=speed)
                combined_wav.unlink()
            else:
                import shutil
                shutil.move(str(combined_wav), str(output_path))

        finally:
            # clean up chunk temp files
            for chunk_wav in chunk_wavs:
                if chunk_wav.exists():
                    chunk_wav.unlink()

        return output_path


# ##################################################################
# get engine
# factory function to get a tts engine by model name and voice
def get_engine(model: str = "qwen", voice: str = DEFAULT_VOICE, voice_description: str | None = None) -> TtsEngine:
    # ##################################################################
    # get engine
    # return the appropriate engine implementation for the given model and voice
    # if voice_description is provided, use VoiceDesignEngine
    # if voice is a registered custom voice name, look up its description or ref_audio

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
        model_name = model if model.startswith("mlx-community/") else "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
        return QwenTtsEngine(model_name=model_name, voice=voice)

    raise ValueError(f"Unknown model: {model}. Supported: qwen")


# ##################################################################
# tts engine module
# provides text-to-speech synthesis with pluggable model backends,
# using mlx-audio for efficient inference on apple silicon
