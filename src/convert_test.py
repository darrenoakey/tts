import pytest
from pathlib import Path
from src.convert import convert_text_to_speech


# ##################################################################
# test convert raises for missing file
# verify conversion fails gracefully when input file does not exist
def test_convert_raises_for_missing_file(tmp_path: Path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        convert_text_to_speech(missing)


# ##################################################################
# test convert raises for empty file
# verify conversion fails gracefully when input file is empty
def test_convert_raises_for_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="Input file is empty"):
        convert_text_to_speech(empty_file)


# ##################################################################
# test convert default output path
# verify default output path is input path with wav extension
def test_convert_default_output_path():
    input_path = Path("/some/path/file.txt")
    expected = Path("/some/path/file.wav")
    assert input_path.with_suffix(".wav") == expected


# ##################################################################
# convert tests
# unit tests for the text-to-speech conversion function
# verifying input validation and output path handling
