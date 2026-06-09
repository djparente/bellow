"""Smoke tests for BellowApp hotkey handler wiring.

Audio I/O and sound effects are mocked out; audio blocks are injected
through the recorder callback.
"""

import unittest
from unittest import mock

import numpy as np

from bellow.main import BellowApp, build_parser
from bellow.recorder import RecorderState
from test_recorder import FakeStream, make_block


class AppTests(unittest.TestCase):
    def setUp(self):
        FakeStream.instances = []
        for target in ('bellow.recorder.sd.InputStream', ):
            patcher = mock.patch(target, FakeStream)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch('bellow.main.play_effect')
        self.play_effect = patcher.start()
        self.addCleanup(patcher.stop)

        args = build_parser().parse_args(['--no-keyboard'])
        self.app = BellowApp(args)
        self.app.transcriber.submit = mock.Mock()
        self.app.recorder.enable()

    def feed(self, seconds: float, value: float = 0.5):
        frames = int(seconds * self.app.recorder.samplerate)
        self.app.recorder._callback(make_block(value, frames), frames, None, None)

    def test_toggle_records_and_submits(self):
        self.app.on_toggle()
        self.assertEqual(self.app.recorder.state, RecorderState.RECORDING)
        self.feed(2.0)
        self.app.on_toggle()
        self.assertEqual(self.app.recorder.state, RecorderState.STANDBY)
        self.app.transcriber.submit.assert_called_once()
        audio = self.app.transcriber.submit.call_args[0][0]
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(len(audio), 32000)

    def test_too_short_recording_is_not_submitted(self):
        self.app.on_toggle()
        self.feed(0.1)
        self.app.on_toggle()
        self.app.transcriber.submit.assert_not_called()

    def test_dump_discards_recording(self):
        self.app.on_toggle()
        self.feed(2.0)
        self.app.on_dump()
        self.assertEqual(self.app.recorder.state, RecorderState.STANDBY)
        self.app.transcriber.submit.assert_not_called()
        self.play_effect.assert_called_with('dump')

    def test_mic_hotkey_releases_and_rearms(self):
        self.app.on_mic()
        self.assertEqual(self.app.recorder.state, RecorderState.OFF)
        self.assertTrue(FakeStream.instances[0].closed)
        self.play_effect.assert_called_with('released')
        self.app.on_mic()
        self.assertEqual(self.app.recorder.state, RecorderState.STANDBY)
        self.play_effect.assert_called_with('armed')

    def test_mic_hotkey_during_recording_discards_and_releases(self):
        self.app.on_toggle()
        self.feed(2.0)
        self.app.on_mic()
        self.assertEqual(self.app.recorder.state, RecorderState.OFF)
        self.app.transcriber.submit.assert_not_called()
        self.play_effect.assert_called_with('dump')

    def test_toggle_while_released_auto_arms_and_records(self):
        self.app.on_mic()  # release
        self.app.on_toggle()
        self.assertEqual(self.app.recorder.state, RecorderState.RECORDING)


if __name__ == '__main__':
    unittest.main()
