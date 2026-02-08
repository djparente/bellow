"""Tests for audio resampling helpers in main.py."""

import numpy as np
import pytest
from bellow.main import _resample_to_16k, WHISPER_SR


class TestResampleTo16k:
    def test_passthrough_at_16k(self):
        """If input is already 16 kHz, return as-is (float32)."""
        x = np.random.randn(16000).astype(np.float64)
        out = _resample_to_16k(x, 16000)
        assert out.dtype == np.float32
        assert out.shape == x.shape
        np.testing.assert_allclose(out, x.astype(np.float32), atol=1e-7)

    def test_downsample_48k_to_16k(self):
        """48 kHz -> 16 kHz: output should have 1/3 the samples."""
        n_in = 48000  # 1 second at 48 kHz
        x = np.sin(2 * np.pi * 440 * np.arange(n_in) / 48000).astype(np.float32)
        out = _resample_to_16k(x, 48000)
        assert out.dtype == np.float32
        # Should be approximately 16000 samples (1 second at 16 kHz)
        assert abs(out.shape[0] - 16000) <= 1

    def test_downsample_44100_to_16k(self):
        """44.1 kHz -> 16 kHz: non-integer ratio resampling."""
        n_in = 44100
        x = np.sin(2 * np.pi * 440 * np.arange(n_in) / 44100).astype(np.float32)
        out = _resample_to_16k(x, 44100)
        assert out.dtype == np.float32
        assert abs(out.shape[0] - 16000) <= 1

    def test_upsample_8k_to_16k(self):
        """8 kHz -> 16 kHz: upsampling doubles sample count."""
        n_in = 8000
        x = np.ones(n_in, dtype=np.float32)
        out = _resample_to_16k(x, 8000)
        assert out.dtype == np.float32
        assert abs(out.shape[0] - 16000) <= 1

    def test_empty_input(self):
        """Empty array should return empty float32 array."""
        x = np.zeros((0,), dtype=np.float32)
        out = _resample_to_16k(x, 48000)
        assert out.dtype == np.float32
        assert out.shape[0] == 0

    def test_very_short_input(self):
        """Very short input (1 sample) at different rate."""
        x = np.array([0.5], dtype=np.float32)
        out = _resample_to_16k(x, 48000)
        assert out.dtype == np.float32
        # With such short input, output may be empty or very short
        assert out.shape[0] >= 0

    def test_output_range_preserved(self):
        """Values in [-1, 1] should stay approximately in that range after resampling."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, size=48000).astype(np.float32)
        out = _resample_to_16k(x, 48000)
        # Allow small overshoot from interpolation filters
        assert out.max() <= 1.5
        assert out.min() >= -1.5
