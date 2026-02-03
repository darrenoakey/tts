import tempfile
from pathlib import Path

import pytest
from src.tts_engine import (
    get_engine,
    QwenTtsEngine,
    TtsEngine,
    QWEN_VOICES,
    DEFAULT_VOICE,
    split_into_chunks,
    is_valid_voice,
    get_all_valid_voices,
    parse_multi_speaker_jsonl,
    validate_multi_speaker_voices,
    group_consecutive_speakers,
)


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
# verify short text returns single chunk when under word limit
def test_split_into_chunks_short_text():
    text = "Hello world. This is a test."
    chunks = split_into_chunks(text, max_words=100)
    assert len(chunks) == 1
    assert chunks[0] == text


# ##################################################################
# test split into chunks with long text
# verify long text is split at sentence boundaries based on word count
def test_split_into_chunks_long_text():
    # each sentence has 2 words, limit to 5 words per chunk
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six."
    chunks = split_into_chunks(text, max_words=5)
    assert len(chunks) == 3
    assert chunks[0] == "Sentence one. Sentence two."  # 4 words
    assert chunks[1] == "Sentence three. Sentence four."  # 4 words
    assert chunks[2] == "Sentence five. Sentence six."  # 4 words


# ##################################################################
# test split into chunks handles different punctuation
# verify splitting works with exclamation and question marks
def test_split_into_chunks_different_punctuation():
    text = "Hello! How are you? I am fine. That is great!"
    # limit to 8 words allows "Hello!"(1) + "How are you?"(3) + "I am fine."(3) = 7 words
    chunks = split_into_chunks(text, max_words=8)
    assert len(chunks) == 2
    assert chunks[0] == "Hello! How are you? I am fine."  # 7 words
    assert chunks[1] == "That is great!"  # 4 words


# ##################################################################
# test split into chunks with empty text
# verify empty text returns empty list
def test_split_into_chunks_empty():
    chunks = split_into_chunks("", max_words=100)
    assert chunks == []


# ##################################################################
# test split into chunks respects sentence boundaries
# verify chunks don't exceed word limit and stop at sentences
def test_split_into_chunks_word_boundary():
    # 10 word sentence followed by 5 word sentence
    text = "This is a longer sentence with exactly ten words in it. Short five word sentence here."
    chunks = split_into_chunks(text, max_words=12)
    assert len(chunks) == 2
    assert chunks[0] == "This is a longer sentence with exactly ten words in it."  # 10 words
    assert chunks[1] == "Short five word sentence here."  # 5 words


# ##################################################################
# test is valid voice with built-in voices
# verify all built-in qwen voices are recognized
def test_is_valid_voice_builtin():
    for voice in QWEN_VOICES:
        assert is_valid_voice(voice) is True


# ##################################################################
# test is valid voice with invalid voice
# verify unknown voices are rejected
def test_is_valid_voice_invalid():
    assert is_valid_voice("nonexistent_voice_xyz") is False
    assert is_valid_voice("") is False
    assert is_valid_voice("INVALID") is False


# ##################################################################
# test is valid voice case insensitive
# verify voice matching is case insensitive for built-in voices
def test_is_valid_voice_case_insensitive():
    assert is_valid_voice("AIDEN") is True
    assert is_valid_voice("Aiden") is True
    assert is_valid_voice("aiden") is True


# ##################################################################
# test get all valid voices
# verify returns both built-in and custom voices
def test_get_all_valid_voices():
    voices = get_all_valid_voices()
    # should include all built-in voices
    for v in QWEN_VOICES:
        assert v in voices
    # should be a non-empty list
    assert len(voices) >= len(QWEN_VOICES)


# ##################################################################
# test parse multi speaker jsonl basic
# verify parsing of valid jsonl input
def test_parse_multi_speaker_jsonl_basic():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"aiden": "Hello there"}\n')
        f.write('{"ryan": "Hi back"}\n')
        f.write('{"aiden": "How are you?"}\n')
        jsonl_path = Path(f.name)

    try:
        segments = parse_multi_speaker_jsonl(jsonl_path)
        assert len(segments) == 3
        assert segments[0] == ("aiden", "Hello there")
        assert segments[1] == ("ryan", "Hi back")
        assert segments[2] == ("aiden", "How are you?")
    finally:
        jsonl_path.unlink()


# ##################################################################
# test parse multi speaker jsonl skips empty lines
# verify empty lines are ignored
def test_parse_multi_speaker_jsonl_empty_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"aiden": "Hello"}\n')
        f.write("\n")
        f.write('{"ryan": "World"}\n')
        f.write("   \n")
        jsonl_path = Path(f.name)

    try:
        segments = parse_multi_speaker_jsonl(jsonl_path)
        assert len(segments) == 2
    finally:
        jsonl_path.unlink()


# ##################################################################
# test parse multi speaker jsonl invalid json
# verify error on malformed json
def test_parse_multi_speaker_jsonl_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"aiden": "Hello"}\n')
        f.write("not valid json\n")
        jsonl_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            parse_multi_speaker_jsonl(jsonl_path)
    finally:
        jsonl_path.unlink()


# ##################################################################
# test parse multi speaker jsonl wrong type
# verify error when line is not object
def test_parse_multi_speaker_jsonl_wrong_type():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('["array", "not", "object"]\n')
        jsonl_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="must be a JSON object"):
            parse_multi_speaker_jsonl(jsonl_path)
    finally:
        jsonl_path.unlink()


# ##################################################################
# test parse multi speaker jsonl multiple keys
# verify error when object has multiple keys
def test_parse_multi_speaker_jsonl_multiple_keys():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"aiden": "Hello", "ryan": "World"}\n')
        jsonl_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="exactly one key"):
            parse_multi_speaker_jsonl(jsonl_path)
    finally:
        jsonl_path.unlink()


# ##################################################################
# test parse multi speaker jsonl file not found
# verify error when file doesn't exist
def test_parse_multi_speaker_jsonl_not_found():
    with pytest.raises(FileNotFoundError):
        parse_multi_speaker_jsonl(Path("/nonexistent/path.jsonl"))


# ##################################################################
# test validate multi speaker voices valid
# verify no error when all voices are valid
def test_validate_multi_speaker_voices_valid():
    segments = [("aiden", "Hello"), ("ryan", "World"), ("vivian", "Hi")]
    # should not raise
    validate_multi_speaker_voices(segments)


# ##################################################################
# test validate multi speaker voices invalid
# verify error listing all invalid voices
def test_validate_multi_speaker_voices_invalid():
    segments = [
        ("aiden", "Hello"),
        ("fake_voice_1", "Text"),
        ("ryan", "World"),
        ("fake_voice_2", "More text"),
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_multi_speaker_voices(segments)

    error_msg = str(exc_info.value)
    assert "fake_voice_1" in error_msg
    assert "fake_voice_2" in error_msg
    assert "Unknown voice(s)" in error_msg


# ##################################################################
# test group consecutive speakers basic
# verify consecutive same-speaker lines are merged
def test_group_consecutive_speakers_basic():
    segments = [
        ("aiden", "Hello"),
        ("aiden", "How are you?"),
        ("ryan", "I'm fine"),
        ("aiden", "Great!"),
    ]
    grouped = group_consecutive_speakers(segments)

    assert len(grouped) == 3
    assert grouped[0] == ("aiden", "Hello How are you?")
    assert grouped[1] == ("ryan", "I'm fine")
    assert grouped[2] == ("aiden", "Great!")


# ##################################################################
# test group consecutive speakers all same
# verify all same-speaker lines become one segment
def test_group_consecutive_speakers_all_same():
    segments = [
        ("aiden", "One"),
        ("aiden", "Two"),
        ("aiden", "Three"),
        ("aiden", "Four"),
        ("aiden", "Five"),
    ]
    grouped = group_consecutive_speakers(segments)

    assert len(grouped) == 1
    assert grouped[0] == ("aiden", "One Two Three Four Five")


# ##################################################################
# test group consecutive speakers alternating
# verify alternating speakers are not merged
def test_group_consecutive_speakers_alternating():
    segments = [
        ("aiden", "A"),
        ("ryan", "B"),
        ("aiden", "C"),
        ("ryan", "D"),
    ]
    grouped = group_consecutive_speakers(segments)

    assert len(grouped) == 4
    assert grouped[0] == ("aiden", "A")
    assert grouped[1] == ("ryan", "B")
    assert grouped[2] == ("aiden", "C")
    assert grouped[3] == ("ryan", "D")


# ##################################################################
# test group consecutive speakers empty
# verify empty input returns empty output
def test_group_consecutive_speakers_empty():
    grouped = group_consecutive_speakers([])
    assert grouped == []


# ##################################################################
# test group consecutive speakers case insensitive
# verify voice name matching is case insensitive
def test_group_consecutive_speakers_case_insensitive():
    segments = [
        ("Aiden", "Hello"),
        ("AIDEN", "World"),
        ("aiden", "Test"),
    ]
    grouped = group_consecutive_speakers(segments)

    assert len(grouped) == 1
    # preserves the case of the first occurrence
    assert grouped[0][0] == "Aiden"
    assert grouped[0][1] == "Hello World Test"


# ##################################################################
# test group consecutive speakers three voices
# verify grouping works with 3+ different speakers
def test_group_consecutive_speakers_three_voices():
    segments = [
        ("aiden", "I'm Aiden"),
        ("aiden", "Nice to meet you"),
        ("ryan", "I'm Ryan"),
        ("vivian", "And I'm Vivian"),
        ("vivian", "Hello everyone"),
        ("aiden", "Welcome all"),
    ]
    grouped = group_consecutive_speakers(segments)

    assert len(grouped) == 4
    assert grouped[0] == ("aiden", "I'm Aiden Nice to meet you")
    assert grouped[1] == ("ryan", "I'm Ryan")
    assert grouped[2] == ("vivian", "And I'm Vivian Hello everyone")
    assert grouped[3] == ("aiden", "Welcome all")


# ##################################################################
# tts engine tests
# unit tests for the tts engine factory and qwen implementation
# verifying correct instantiation, voice selection, and interface compliance
