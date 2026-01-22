import subprocess
import tempfile
import torch
import soundfile as sf
from pathlib import Path
from abc import ABC, abstractmethod


# available voices for qwen3-tts customvoice model
QWEN_VOICES = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
DEFAULT_VOICE = "aiden"


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
# implementation using qwen3-tts models for high quality speech synthesis
class QwenTtsEngine(TtsEngine):

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", voice: str = DEFAULT_VOICE):
        # ##################################################################
        # init
        # load the qwen tts model with specified variant and voice
        self.model_name = model_name
        if voice not in QWEN_VOICES:
            raise ValueError(f"Unknown voice: {voice}. Supported: {', '.join(QWEN_VOICES)}")
        self.voice = voice
        self._model = None

    def _get_device_and_dtype(self) -> tuple[str, torch.dtype]:
        # ##################################################################
        # get device and dtype
        # determine best available device - cuda preferred, mps experimental, cpu fallback
        if torch.cuda.is_available():
            return "cuda:0", torch.bfloat16
        if torch.backends.mps.is_available():
            # mps support for qwen-tts is experimental and may not work
            # fall back to cpu for reliability
            return "cpu", torch.float32
        return "cpu", torch.float32

    def _get_model(self):
        # ##################################################################
        # get model
        # lazy load model with flash attention shim for non-cuda systems
        if self._model is None:
            from src.flash_attn_shim import install_flash_attn_shim
            install_flash_attn_shim()
            from qwen_tts import Qwen3TTSModel
            device, dtype = self._get_device_and_dtype()
            kwargs = {
                "device_map": device,
                "dtype": dtype,
            }
            if device.startswith("cuda"):
                try:
                    kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    pass
            self._model = Qwen3TTSModel.from_pretrained(self.model_name, **kwargs)
        return self._model

    def synthesize(self, text: str, output_path: Path, language: str = "English") -> Path:
        # ##################################################################
        # synthesize
        # generate speech from text using qwen model and save as mp3
        model = self._get_model()
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=language,
            speaker=self.voice,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == ".mp3":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = Path(tmp.name)
            sf.write(str(tmp_wav), wavs[0], sr)
            convert_wav_to_mp3(tmp_wav, output_path)
            tmp_wav.unlink()
        else:
            sf.write(str(output_path), wavs[0], sr)
        return output_path


# ##################################################################
# get engine
# factory function to get a tts engine by model name and voice
def get_engine(model: str = "qwen", voice: str = DEFAULT_VOICE) -> TtsEngine:
    # ##################################################################
    # get engine
    # return the appropriate engine implementation for the given model and voice
    if model == "qwen" or model.startswith("Qwen/"):
        model_name = model if model.startswith("Qwen/") else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        return QwenTtsEngine(model_name=model_name, voice=voice)
    raise ValueError(f"Unknown model: {model}. Supported: qwen")


# ##################################################################
# tts engine module
# provides text-to-speech synthesis with pluggable model backends,
# currently supporting qwen3-tts for high quality speech generation
