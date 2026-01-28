import gc
import re
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from abc import ABC, abstractmethod


# available voices for qwen3-tts customvoice model (mlx-audio)
# english: aiden, ryan
# chinese: vivian, serena, uncle_fu, dylan, eric
QWEN_VOICES = ["aiden", "ryan", "vivian", "serena", "uncle_fu", "dylan", "eric"]
DEFAULT_VOICE = "aiden"

# chunk size for processing long texts (in sentences)
# smaller chunks = more stable memory but slightly more overhead
DEFAULT_CHUNK_SENTENCES = 3

# sample rate for qwen3-tts models
SAMPLE_RATE = 12000


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
# use ffmpeg to compress wav audio to mp3 format
def convert_wav_to_mp3(wav_path: Path, mp3_path: Path) -> Path:
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-qscale:a", "2",
        str(mp3_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    return mp3_path


# ##################################################################
# concatenate wav files
# use ffmpeg to concatenate multiple wav files into one
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
    def synthesize(self, text: str, output_path: Path, language: str = "English") -> Path:
        # ##################################################################
        # synthesize
        # convert text to speech and write to output_path, returning the path
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

    def synthesize(self, text: str, output_path: Path, language: str = "English") -> Path:
        # ##################################################################
        # synthesize
        # generate speech from text using qwen model and save as mp3
        # processes text in chunks to avoid memory spikes on long inputs
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
                sf.write(str(chunk_wav), audio_np, SAMPLE_RATE)
                chunk_wavs.append(chunk_wav)

                # free memory from this chunk
                del results, audio, audio_np
                gc.collect()
                mx.clear_cache()

            # concatenate all chunk wavs into final output
            if output_path.suffix.lower() == ".mp3":
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    combined_wav = Path(tmp.name)
                concatenate_wav_files(chunk_wavs, combined_wav)
                convert_wav_to_mp3(combined_wav, output_path)
                combined_wav.unlink()
            else:
                concatenate_wav_files(chunk_wavs, output_path)

        finally:
            # clean up chunk temp files
            for chunk_wav in chunk_wavs:
                if chunk_wav.exists():
                    chunk_wav.unlink()

        return output_path


# ##################################################################
# get engine
# factory function to get a tts engine by model name and voice
def get_engine(model: str = "qwen", voice: str = DEFAULT_VOICE) -> TtsEngine:
    # ##################################################################
    # get engine
    # return the appropriate engine implementation for the given model and voice
    if model == "qwen" or model.startswith("mlx-community/"):
        model_name = model if model.startswith("mlx-community/") else "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
        return QwenTtsEngine(model_name=model_name, voice=voice)
    raise ValueError(f"Unknown model: {model}. Supported: qwen")


# ##################################################################
# tts engine module
# provides text-to-speech synthesis with pluggable model backends,
# using mlx-audio for efficient inference on apple silicon
