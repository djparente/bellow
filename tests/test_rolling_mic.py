"""Tests for the RollingMic class (mocking sounddevice)."""

import time
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from threading import Event

from bellow.main import RollingMic, pause_recording


@pytest.fixture(autouse=True)
def reset_pause():
    """Ensure pause_recording is cleared before each test."""
    pause_recording.clear()
    yield
    pause_recording.clear()


def _make_rolling_mic(prebuf_seconds=1.0, idle_timeout_s=30.0):
    """Create a RollingMic with mocked sounddevice, simulating a working stream."""
    with patch("bellow.main._find_working_samplerate", return_value=16000), \
         patch("bellow.main.sd") as mock_sd:
        # Mock InputStream that captures the callback
        mock_stream = MagicMock()
        mock_stream.start = MagicMock()
        mock_stream.stop = MagicMock()
        mock_stream.close = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        mic = RollingMic(device=None, prebuf_seconds=prebuf_seconds, idle_timeout_s=idle_timeout_s)
        return mic, mock_stream


class TestRollingMicInit:
    def test_creates_stream(self):
        mic, mock_stream = _make_rolling_mic()
        assert mic.stream is not None
        mock_stream.start.assert_called_once()
        assert mic.sr_in == 16000
        mic.close()

    def test_prebuf_seconds_clamped(self):
        mic, _ = _make_rolling_mic(prebuf_seconds=0.01)
        assert mic.prebuf_seconds == 0.1  # minimum is 0.1
        mic.close()

    def test_prebuf_limit_calculated(self):
        mic, _ = _make_rolling_mic(prebuf_seconds=1.0)
        assert mic.prebuf_limit == 16000  # 1.0s * 16000 Hz
        mic.close()


class TestRollingMicCallback:
    def test_prebuffer_fills(self):
        mic, _ = _make_rolling_mic(prebuf_seconds=1.0)

        # Simulate callback with audio data
        chunk = np.zeros((1600, 1), dtype=np.int16)  # 0.1s at 16kHz
        mic._callback(chunk, 1600, None, None)

        assert mic.prebuf_samples == 1600
        assert len(mic.prebuf) == 1
        mic.close()

    def test_prebuffer_rolls(self):
        mic, _ = _make_rolling_mic(prebuf_seconds=0.5)
        # prebuf_limit = 8000 samples

        # Add 1 second worth of data in chunks (exceeds 0.5s buffer)
        for _ in range(10):
            chunk = np.zeros((1600, 1), dtype=np.int16)
            mic._callback(chunk, 1600, None, None)

        # Should have rolled off old data, keeping at most ~8000 samples
        assert mic.prebuf_samples <= 8000 + 1600  # allow one chunk overshoot
        mic.close()

    def test_session_frames_captured(self):
        mic, _ = _make_rolling_mic()
        mic.session_active = True

        chunk = np.ones((800, 1), dtype=np.int16)
        mic._callback(chunk, 800, None, None)

        assert len(mic.session_frames) == 1
        np.testing.assert_array_equal(mic.session_frames[0], chunk)
        mic.close()

    def test_paused_session_skips_frames(self):
        mic, _ = _make_rolling_mic()
        mic.session_active = True
        pause_recording.set()

        chunk = np.ones((800, 1), dtype=np.int16)
        mic._callback(chunk, 800, None, None)

        # Prebuffer should still get the data
        assert mic.prebuf_samples == 800
        # But session frames should be empty
        assert len(mic.session_frames) == 0
        mic.close()


class TestRollingMicSession:
    def test_start_session_with_preroll(self):
        mic, _ = _make_rolling_mic(prebuf_seconds=1.0)

        # Fill prebuffer with known data
        chunk = np.full((16000, 1), 42, dtype=np.int16)  # 1 second
        mic._callback(chunk, 16000, None, None)

        # Start session requesting 0.5s preroll
        mic.start_session(preroll_s=0.5)
        assert mic.session_active is True

        # Session frames should have the preroll (last 0.5s = 8000 samples)
        assert len(mic.session_frames) == 1
        assert mic.session_frames[0].shape[0] == 8000
        mic.close()

    def test_start_session_no_preroll(self):
        mic, _ = _make_rolling_mic()
        mic.start_session(preroll_s=0.0)
        assert mic.session_active is True
        assert len(mic.session_frames) == 0
        mic.close()

    def test_stop_session_returns_data(self):
        mic, _ = _make_rolling_mic()
        mic.start_session(preroll_s=0.0)

        chunk = np.full((800, 1), 100, dtype=np.int16)
        mic._callback(chunk, 800, None, None)

        result = mic.stop_session()
        assert result is not None
        assert result.shape[0] == 800
        assert mic.session_active is False
        mic.close()

    def test_stop_session_empty_returns_none(self):
        mic, _ = _make_rolling_mic()
        mic.start_session(preroll_s=0.0)
        result = mic.stop_session()
        assert result is None
        mic.close()

    def test_stop_session_concatenates_frames(self):
        mic, _ = _make_rolling_mic()
        mic.start_session(preroll_s=0.0)

        for i in range(5):
            chunk = np.full((160, 1), i, dtype=np.int16)
            mic._callback(chunk, 160, None, None)

        result = mic.stop_session()
        assert result.shape[0] == 800  # 5 * 160
        mic.close()


class TestRollingMicIdleTimeout:
    def test_idle_timer_starts_on_stop(self):
        mic, _ = _make_rolling_mic(idle_timeout_s=0.1)
        mic.start_session(preroll_s=0.0)
        mic.stop_session()

        # Timer should be set
        assert mic._idle_timer is not None
        mic.close()

    def test_idle_timeout_closes_stream(self):
        mic, mock_stream = _make_rolling_mic(idle_timeout_s=0.2)
        mic.start_session(preroll_s=0.0)
        mic.stop_session()

        # Wait for idle timeout to fire
        time.sleep(0.4)

        assert mic.stream is None
        mock_stream.stop.assert_called()
        mock_stream.close.assert_called()
        mic.close()

    def test_no_timeout_when_zero(self):
        mic, _ = _make_rolling_mic(idle_timeout_s=0)
        mic.start_session(preroll_s=0.0)
        mic.stop_session()

        assert mic._idle_timer is None
        assert mic.stream is not None
        mic.close()

    def test_session_cancels_idle_timer(self):
        mic, _ = _make_rolling_mic(idle_timeout_s=1.0)
        mic.start_session(preroll_s=0.0)
        mic.stop_session()
        assert mic._idle_timer is not None

        # Starting a new session should cancel the timer
        mic.start_session(preroll_s=0.0)
        assert mic._idle_timer is None
        assert mic.stream is not None
        mic.close()

    def test_reopen_after_idle_close(self):
        mic, mock_stream = _make_rolling_mic(idle_timeout_s=0.1)
        mic.start_session(preroll_s=0.0)
        mic.stop_session()

        time.sleep(0.3)
        assert mic.stream is None

        # Reopen by starting a new session
        with patch("bellow.main._find_working_samplerate", return_value=16000), \
             patch("bellow.main.sd") as mock_sd2:
            new_stream = MagicMock()
            mock_sd2.InputStream.return_value = new_stream
            mic.start_session(preroll_s=0.0)

        assert mic.stream is not None
        assert mic.session_active is True
        mic.close()


class TestRollingMicClose:
    def test_close_permanent(self):
        mic, mock_stream = _make_rolling_mic()
        mic.close()
        assert mic.stream is None
        assert mic._closed_permanently is True
        mock_stream.stop.assert_called()
        mock_stream.close.assert_called()

    def test_close_prevents_reopen(self):
        mic, _ = _make_rolling_mic()
        mic.close()

        with patch("bellow.main._find_working_samplerate", return_value=16000), \
             patch("bellow.main.sd"):
            mic._open_stream()

        assert mic.stream is None  # should not reopen

    def test_is_stream_open(self):
        mic, _ = _make_rolling_mic()
        assert mic.is_stream_open is True
        mic.close()
        assert mic.is_stream_open is False
