"""Tests for the transcription backend abstraction."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from bellow.transcription import (
    _seconds_of,
    _split_audio,
    _extract_hf_text,
    _hf_name_to_fw_size,
    create_backend,
    HuggingFaceBackend,
    FasterWhisperBackend,
    WHISPER_SR,
)


class TestSecondsOf:
    def test_one_second(self):
        x = np.zeros(WHISPER_SR, dtype=np.float32)
        assert abs(_seconds_of(x) - 1.0) < 0.001

    def test_empty(self):
        x = np.zeros(0, dtype=np.float32)
        assert _seconds_of(x) == 0.0

    def test_none(self):
        assert _seconds_of(None) == 0.0

    def test_half_second(self):
        x = np.zeros(WHISPER_SR // 2, dtype=np.float32)
        assert abs(_seconds_of(x) - 0.5) < 0.001


class TestSplitAudio:
    def test_no_split_short_audio(self):
        x = np.zeros(WHISPER_SR * 10, dtype=np.float32)  # 10s
        parts = _split_audio(x, max_seconds=30.0)
        assert len(parts) == 1
        assert parts[0].shape[0] == x.shape[0]

    def test_split_long_audio(self):
        x = np.zeros(WHISPER_SR * 60, dtype=np.float32)  # 60s
        parts = _split_audio(x, max_seconds=29.9)
        # Should split into 3 parts (29.9s + 29.9s + ~0.2s)
        assert len(parts) == 3
        total = sum(p.shape[0] for p in parts)
        assert total == x.shape[0]

    def test_split_zero_max(self):
        x = np.zeros(WHISPER_SR, dtype=np.float32)
        parts = _split_audio(x, max_seconds=0)
        assert len(parts) == 1

    def test_split_empty_audio(self):
        x = np.zeros(0, dtype=np.float32)
        parts = _split_audio(x, max_seconds=10.0)
        assert len(parts) == 1
        assert parts[0].shape[0] == 0


class TestExtractHfText:
    def test_simple_text(self):
        assert _extract_hf_text({"text": " Hello world"}) == "Hello world"

    def test_none_text(self):
        assert _extract_hf_text({"text": None}) == ""

    def test_chunks(self):
        result = {"chunks": [
            {"text": " Hello"},
            {"text": " world"},
        ]}
        assert _extract_hf_text(result) == "Hello world"

    def test_empty_dict(self):
        assert _extract_hf_text({}) == ""

    def test_non_dict(self):
        assert _extract_hf_text("not a dict") == ""

    def test_none_input(self):
        assert _extract_hf_text(None) == ""

    def test_leading_whitespace_stripped(self):
        assert _extract_hf_text({"text": "   lots of space"}) == "lots of space"


class TestHfNameToFwSize:
    def test_whisper_medium(self):
        assert _hf_name_to_fw_size("openai/whisper-medium") == "medium"

    def test_whisper_large_v2(self):
        assert _hf_name_to_fw_size("openai/whisper-large-v2") == "large-v2"

    def test_whisper_tiny(self):
        assert _hf_name_to_fw_size("openai/whisper-tiny") == "tiny"

    def test_bare_size(self):
        assert _hf_name_to_fw_size("medium") == "medium"

    def test_custom_model_path(self):
        assert _hf_name_to_fw_size("my-org/my-custom-model") == "my-custom-model"

    def test_whitespace_stripped(self):
        assert _hf_name_to_fw_size("  openai/whisper-small  ") == "small"


class TestCreateBackend:
    def test_huggingface(self):
        backend = create_backend("huggingface")
        assert isinstance(backend, HuggingFaceBackend)

    def test_hf_alias(self):
        backend = create_backend("hf")
        assert isinstance(backend, HuggingFaceBackend)

    def test_transformers_alias(self):
        backend = create_backend("transformers")
        assert isinstance(backend, HuggingFaceBackend)

    def test_unknown_falls_back(self):
        backend = create_backend("nonexistent")
        assert isinstance(backend, HuggingFaceBackend)

    @patch.dict("sys.modules", {"faster_whisper": MagicMock()})
    def test_faster_whisper_when_available(self):
        backend = create_backend("faster-whisper")
        assert isinstance(backend, FasterWhisperBackend)

    def test_faster_whisper_fallback_when_missing(self):
        """If faster-whisper is not installed, should fall back to HuggingFace."""
        # Ensure faster_whisper is not importable
        with patch.dict("sys.modules", {"faster_whisper": None}):
            backend = create_backend("faster-whisper")
            assert isinstance(backend, HuggingFaceBackend)

    def test_fw_alias(self):
        with patch.dict("sys.modules", {"faster_whisper": MagicMock()}):
            backend = create_backend("fw")
            assert isinstance(backend, FasterWhisperBackend)


class TestHuggingFaceBackendTranscribe:
    def test_transcribe_without_loading_returns_empty(self):
        backend = HuggingFaceBackend()
        audio = np.zeros(WHISPER_SR, dtype=np.float32)
        result = backend.transcribe(audio)
        assert result == ""

    def test_transcribe_calls_pipeline(self):
        backend = HuggingFaceBackend()
        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": " Hello world"}
        backend._pipe = mock_pipe

        audio = np.zeros(WHISPER_SR * 5, dtype=np.float32)  # 5 seconds
        result = backend.transcribe(audio, timestamps_mode="off")
        assert result == "Hello world"
        mock_pipe.assert_called_once()

    def test_transcribe_long_audio_splits(self):
        """Audio > no_ts_max_seconds with timestamps=off should be split."""
        backend = HuggingFaceBackend()
        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": " chunk"}
        backend._pipe = mock_pipe

        audio = np.zeros(WHISPER_SR * 60, dtype=np.float32)  # 60 seconds
        result = backend.transcribe(audio, timestamps_mode="off", no_ts_max_seconds=29.9)
        assert "chunk" in result
        # Should have been called multiple times (60s / 29.9s = 3 chunks)
        assert mock_pipe.call_count == 3

    def test_transcribe_auto_timestamps_short(self):
        """Short audio with auto timestamps should NOT use timestamps."""
        backend = HuggingFaceBackend()
        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": " short"}
        backend._pipe = mock_pipe

        audio = np.zeros(WHISPER_SR * 10, dtype=np.float32)
        result = backend.transcribe(audio, timestamps_mode="auto")
        assert result == "short"
        # Should have been called without return_timestamps
        call_kwargs = mock_pipe.call_args[1]
        assert "return_timestamps" not in call_kwargs

    def test_transcribe_auto_timestamps_long(self):
        """Long audio (>=30s) with auto timestamps should use timestamps."""
        backend = HuggingFaceBackend()
        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": " long audio"}
        backend._pipe = mock_pipe

        audio = np.zeros(WHISPER_SR * 35, dtype=np.float32)
        result = backend.transcribe(audio, timestamps_mode="auto")
        assert result == "long audio"
        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs.get("return_timestamps") is True
