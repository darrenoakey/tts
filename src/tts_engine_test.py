import pytest
from src.tts_engine import get_engine, QwenTtsEngine, TtsEngine, QWEN_VOICES, DEFAULT_VOICE, split_into_chunks


# ##################################################################
# test get engine returns qwen for default
# verify factory returns qwen engine when no model specified
def test_get_engine_returns_qwen_for_default():
    engine = get_engine()
    assert isinstance(engine, QwenTtsEngine)


# ##################################################################
# test get engine returns qwen for explicit qwen
# verify factory returns qwen engine when explicitly requested
def test_get_engine_returns_qwen_for_explicit_qwen():
    engine = get_engine("qwen")
    assert isinstance(engine, QwenTtsEngine)


# ##################################################################
# test get engine raises for unknown model
# verify factory raises valueerror for unsupported models
def test_get_engine_raises_for_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        get_engine("unknown_model")


# ##################################################################
# test qwen engine has correct model name
# verify qwen engine stores the expected model identifier
def test_qwen_engine_has_correct_model_name():
    engine = QwenTtsEngine()
    assert "Qwen3-TTS" in engine.model_name


# ##################################################################
# test qwen engine accepts custom model name
# verify qwen engine allows overriding the model name
def test_qwen_engine_accepts_custom_model_name():
    engine = QwenTtsEngine(model_name="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16")
    assert engine.model_name == "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"


# ##################################################################
# test qwen engine default voice
# verify qwen engine uses aiden as default voice
def test_qwen_engine_default_voice():
    engine = QwenTtsEngine()
    assert engine.voice == DEFAULT_VOICE
    assert engine.voice == "aiden"


# ##################################################################
# test qwen engine accepts valid voice
# verify qwen engine allows setting any supported voice
def test_qwen_engine_accepts_valid_voice():
    for voice in QWEN_VOICES:
        engine = QwenTtsEngine(voice=voice)
        assert engine.voice == voice


# ##################################################################
# test qwen engine rejects invalid voice
# verify qwen engine raises error for unsupported voices
def test_qwen_engine_rejects_invalid_voice():
    with pytest.raises(ValueError, match="Unknown voice"):
        QwenTtsEngine(voice="neil")


# ##################################################################
# test get engine with voice
# verify factory passes voice to engine
def test_get_engine_with_voice():
    engine = get_engine("qwen", voice="vivian")
    assert engine.voice == "vivian"


# ##################################################################
# test tts engine interface
# verify tts engine is abstract and defines required methods
def test_tts_engine_interface():
    assert hasattr(TtsEngine, "synthesize")
    with pytest.raises(TypeError):
        TtsEngine()


# ##################################################################
# test split into chunks with short text
# verify short text returns single chunk
def test_split_into_chunks_short_text():
    text = "Hello world. This is a test."
    chunks = split_into_chunks(text, sentences_per_chunk=5)
    assert len(chunks) == 1
    assert chunks[0] == text


# ##################################################################
# test split into chunks with long text
# verify long text is split into multiple chunks
def test_split_into_chunks_long_text():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six."
    chunks = split_into_chunks(text, sentences_per_chunk=2)
    assert len(chunks) == 3
    assert chunks[0] == "Sentence one. Sentence two."
    assert chunks[1] == "Sentence three. Sentence four."
    assert chunks[2] == "Sentence five. Sentence six."


# ##################################################################
# test split into chunks handles different punctuation
# verify splitting works with exclamation and question marks
def test_split_into_chunks_different_punctuation():
    text = "Hello! How are you? I am fine. That is great!"
    chunks = split_into_chunks(text, sentences_per_chunk=2)
    assert len(chunks) == 2
    assert chunks[0] == "Hello! How are you?"
    assert chunks[1] == "I am fine. That is great!"


# ##################################################################
# test split into chunks with empty text
# verify empty text returns empty list
def test_split_into_chunks_empty():
    chunks = split_into_chunks("", sentences_per_chunk=5)
    assert chunks == []


# ##################################################################
# tts engine tests
# unit tests for the tts engine factory and qwen implementation
# verifying correct instantiation, voice selection, and interface compliance
