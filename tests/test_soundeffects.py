"""Tests for the soundeffects module."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from bellow.soundeffects import effects, AudioData, play_effect, read_effect


class TestAudioData:
    def test_namedtuple_fields(self):
        ad = AudioData(data=np.zeros(10), samplerate=44100)
        assert ad.samplerate == 44100
        assert ad.data.shape == (10,)


class TestReadEffect:
    def test_read_valid_file(self, tmp_path):
        """Read a WAV file into the effects dict."""
        import soundfile as sf
        wav_path = tmp_path / "test.wav"
        data = np.random.randn(1000).astype(np.float32)
        sf.write(str(wav_path), data, 16000)

        read_effect(str(wav_path), "test_effect")
        assert "test_effect" in effects
        assert effects["test_effect"].samplerate == 16000
        assert effects["test_effect"].data.shape[0] == 1000

        # Cleanup
        del effects["test_effect"]

    def test_read_nonexistent_file(self):
        """Should log a warning and not crash."""
        read_effect("/nonexistent/path.wav", "missing")
        assert "missing" not in effects


class TestPlayEffect:
    @patch("bellow.soundeffects.sd")
    def test_play_existing_effect(self, mock_sd):
        effects["test_play"] = AudioData(data=np.zeros(100), samplerate=16000)
        play_effect("test_play", blocking=False)
        mock_sd.play.assert_called_once()
        del effects["test_play"]

    @patch("bellow.soundeffects.sd")
    def test_play_blocking(self, mock_sd):
        effects["test_block"] = AudioData(data=np.zeros(100), samplerate=16000)
        play_effect("test_block", blocking=True)
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()
        del effects["test_block"]

    @patch("bellow.soundeffects.sd")
    def test_play_missing_effect(self, mock_sd):
        """Playing a non-existent effect should not crash."""
        play_effect("nonexistent_effect", blocking=False)
        mock_sd.play.assert_not_called()
