"""Unit tests for the Recorder state machine and ring buffer.

The PortAudio stream is replaced with a fake so these tests run without
audio hardware; audio blocks are injected by calling the recorder's
callback directly, exactly as PortAudio would.
"""

import unittest
from unittest import mock

import numpy as np

from bellow.recorder import Recorder, RecorderState, _resample


class FakeStream:
    instances = []

    def __init__(self, samplerate=None, channels=None, dtype=None, device=None, callback=None):
        self.samplerate = samplerate
        self.callback = callback
        self.started = False
        self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class FakeStatus:
    input_overflow = True


def make_block(value: float, frames: int) -> np.ndarray:
    return np.full((frames, 1), value, dtype=np.float32)


class RecorderTests(unittest.TestCase):
    def setUp(self):
        FakeStream.instances = []
        patcher = mock.patch('bellow.recorder.sd.InputStream', FakeStream)
        patcher.start()
        self.addCleanup(patcher.stop)

    def feed(self, recorder, value, frames=100):
        recorder._callback(make_block(value, frames), frames, None, None)

    def test_initial_state_is_off(self):
        rec = Recorder()
        self.assertEqual(rec.state, RecorderState.OFF)

    def test_enable_opens_stream_and_enters_standby(self):
        rec = Recorder()
        rec.enable()
        self.assertEqual(rec.state, RecorderState.STANDBY)
        self.assertTrue(FakeStream.instances[0].started)

    def test_preroll_ring_buffer_trims_to_window(self):
        rec = Recorder(samplerate=1000, preroll_seconds=0.5)  # 500-sample window
        rec.enable()
        for i in range(10):
            self.feed(rec, float(i), frames=100)
        rec.start_recording()
        audio = rec.stop_recording()
        # Only the last 5 blocks (500 samples) should have been kept as pre-roll
        self.assertEqual(len(audio), 500)
        self.assertEqual(audio[0], 5.0)
        self.assertEqual(audio[-1], 9.0)

    def test_recording_includes_preroll_and_new_audio(self):
        rec = Recorder(samplerate=1000, preroll_seconds=0.2)
        rec.enable()
        self.feed(rec, 1.0, frames=200)   # pre-roll
        rec.start_recording()
        self.feed(rec, 2.0, frames=300)   # live speech
        audio = rec.stop_recording()
        self.assertEqual(len(audio), 500)
        np.testing.assert_array_equal(audio[:200], np.full(200, 1.0, dtype=np.float32))
        np.testing.assert_array_equal(audio[200:], np.full(300, 2.0, dtype=np.float32))
        self.assertEqual(rec.state, RecorderState.STANDBY)

    def test_zero_preroll_keeps_no_standby_audio(self):
        rec = Recorder(samplerate=1000, preroll_seconds=0.0)
        rec.enable()
        self.feed(rec, 1.0, frames=400)
        rec.start_recording()
        self.feed(rec, 2.0, frames=100)
        audio = rec.stop_recording()
        self.assertEqual(len(audio), 100)

    def test_start_recording_from_off_opens_stream(self):
        rec = Recorder()
        rec.start_recording()
        self.assertEqual(rec.state, RecorderState.RECORDING)
        self.assertTrue(FakeStream.instances[0].started)

    def test_stop_without_recording_returns_none(self):
        rec = Recorder()
        rec.enable()
        self.assertIsNone(rec.stop_recording())

    def test_stop_with_no_audio_returns_none(self):
        rec = Recorder()
        rec.start_recording()
        self.assertIsNone(rec.stop_recording())

    def test_cancel_discards_audio(self):
        rec = Recorder(samplerate=1000)
        rec.start_recording()
        self.feed(rec, 1.0)
        rec.cancel_recording()
        self.assertEqual(rec.state, RecorderState.STANDBY)
        rec.start_recording()
        self.feed(rec, 2.0, frames=50)
        audio = rec.stop_recording()
        self.assertEqual(len(audio), 50)
        self.assertEqual(audio[0], 2.0)

    def test_disable_releases_stream_and_reports_discarded_recording(self):
        rec = Recorder()
        rec.start_recording()
        self.feed(rec, 1.0)
        self.assertTrue(rec.disable())
        self.assertEqual(rec.state, RecorderState.OFF)
        self.assertTrue(FakeStream.instances[0].closed)
        # Disabling from standby reports no discarded recording
        rec.enable()
        self.assertFalse(rec.disable())

    def test_double_start_is_idempotent(self):
        rec = Recorder(samplerate=1000)
        rec.start_recording()
        self.feed(rec, 1.0, frames=100)
        rec.start_recording()  # should not clear the in-progress recording
        audio = rec.stop_recording()
        self.assertEqual(len(audio), 100)

    def test_overflows_are_counted(self):
        rec = Recorder(samplerate=1000)
        rec.start_recording()
        rec._callback(make_block(1.0, 100), 100, None, FakeStatus())
        with self.assertLogs('bellow.recorder', level='WARNING') as cm:
            rec.stop_recording()
        self.assertTrue(any('overflowed 1 time' in m for m in cm.output))

    def test_resample_changes_length_and_preserves_dtype(self):
        audio = np.sin(np.linspace(0, 10, 480)).astype(np.float32)
        out = _resample(audio, 48000, 16000)
        self.assertEqual(len(out), 160)
        self.assertEqual(out.dtype, np.float32)
        self.assertIs(_resample(audio, 16000, 16000), audio)


if __name__ == '__main__':
    unittest.main()
