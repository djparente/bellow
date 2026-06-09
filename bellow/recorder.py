"""Audio capture for Bellow.

The Recorder is a small state machine around a single, persistent,
callback-driven sounddevice.InputStream:

    OFF       -- no stream is open; the microphone is fully released and
                 available to other applications (e.g. Zoom).
    STANDBY   -- the stream is open and the callback keeps the last
                 `preroll_seconds` of audio in a ring buffer, so that the
                 moment a recording starts no speech has been lost to
                 stream start-up latency.
    RECORDING -- the callback appends every block to the active recording.

Audio is captured with a PortAudio callback rather than blocking reads so
that no samples are dropped if the Python control thread is briefly busy;
overflows reported by PortAudio are counted and logged instead of being
silently ignored.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from enum import Enum

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class RecorderState(Enum):
    OFF = 'off'
    STANDBY = 'standby'
    RECORDING = 'recording'


class Recorder:
    """Persistent-stream microphone recorder with a pre-roll ring buffer."""

    def __init__(self, device: int | str | None = None, samplerate: int = 16000,
                 preroll_seconds: float = 1.0):
        """
        :param device: sounddevice input device index or name; None for the default input
        :param samplerate: desired output sample rate (Whisper expects 16000); if the
            device cannot open at this rate, its default rate is used and the audio is
            resampled when a recording is finalized
        :param preroll_seconds: how much standby audio to keep and prepend to each
            recording; 0 disables pre-roll
        """
        self._device = device
        self._target_samplerate = int(samplerate)
        self._preroll_seconds = max(0.0, float(preroll_seconds))

        self._lock = threading.Lock()
        self._state = RecorderState.OFF
        self._stream: sd.InputStream | None = None
        self._stream_samplerate = self._target_samplerate

        # Ring buffer of recent standby blocks (pre-roll) and the active recording
        self._pre_blocks: deque[np.ndarray] = deque()
        self._pre_samples = 0
        self._rec_blocks: list[np.ndarray] = []
        self._overflows = 0

    @property
    def state(self) -> RecorderState:
        with self._lock:
            return self._state

    @property
    def samplerate(self) -> int:
        """The sample rate of audio returned by stop_recording()."""
        return self._target_samplerate

    # -- state transitions -------------------------------------------------

    def enable(self) -> None:
        """Open the input stream and enter STANDBY. No-op unless OFF.

        Raises an exception if the audio device cannot be opened.
        """
        with self._lock:
            if self._state != RecorderState.OFF:
                return
            self._open_stream_locked()
            self._state = RecorderState.STANDBY
        logger.info('Microphone armed (standby, %.2fs pre-roll)', self._preroll_seconds)

    def disable(self) -> bool:
        """Close the stream and release the microphone from any state.

        :return: True if an in-progress recording was discarded
        """
        with self._lock:
            was_recording = self._state == RecorderState.RECORDING
            stream, self._stream = self._stream, None
            self._state = RecorderState.OFF
            self._clear_buffers_locked()

        # Close outside the lock: stream.close() waits for the callback to
        # return, and the callback acquires the lock.
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                logger.warning('Error while closing input stream: %s', e)
            logger.info('Microphone released')
        return was_recording

    def start_recording(self) -> None:
        """Begin recording, prepending any pre-roll audio. If the recorder is
        OFF, the stream is opened first (in that case no pre-roll exists).

        Raises an exception if the audio device cannot be opened.
        """
        with self._lock:
            if self._state == RecorderState.RECORDING:
                return
            if self._state == RecorderState.OFF:
                self._open_stream_locked()
            # Seed the recording with the pre-roll ring buffer
            self._rec_blocks = list(self._pre_blocks)
            self._pre_blocks.clear()
            self._pre_samples = 0
            self._overflows = 0
            self._state = RecorderState.RECORDING
        logger.info('Recording started')

    def stop_recording(self) -> np.ndarray | None:
        """Stop recording and return the captured audio as float32 mono at
        `samplerate`, or None if nothing was being recorded. The recorder
        returns to STANDBY (the stream stays open; call disable() to release
        the microphone).
        """
        with self._lock:
            if self._state != RecorderState.RECORDING:
                return None
            blocks, self._rec_blocks = self._rec_blocks, []
            overflows = self._overflows
            stream_rate = self._stream_samplerate
            self._state = RecorderState.STANDBY

        if overflows:
            logger.warning('Input overflowed %d time(s) during this recording; '
                           'some audio may be missing', overflows)

        if not blocks:
            logger.info('Recording stopped (no audio captured)')
            return None

        audio = np.concatenate(blocks)
        if stream_rate != self._target_samplerate:
            audio = _resample(audio, stream_rate, self._target_samplerate)
        logger.info('Recording stopped (%.2fs of audio)', len(audio) / self._target_samplerate)
        return audio

    def cancel_recording(self) -> None:
        """Discard any in-progress recording and return to STANDBY."""
        with self._lock:
            if self._state != RecorderState.RECORDING:
                return
            self._rec_blocks = []
            self._state = RecorderState.STANDBY
        logger.info('Recording discarded')

    # -- internals ----------------------------------------------------------

    def _open_stream_locked(self) -> None:
        """Open and start the input stream. Tries the target sample rate first
        and falls back to the device default (resampling later) if the device
        refuses it. Must be called with the lock held and no stream open.
        """
        try:
            stream = sd.InputStream(samplerate=self._target_samplerate, channels=1,
                                    dtype='float32', device=self._device,
                                    callback=self._callback)
            self._stream_samplerate = self._target_samplerate
        except sd.PortAudioError:
            default_rate = int(sd.query_devices(self._device, 'input')['default_samplerate'])
            logger.warning('Device does not support %d Hz; capturing at %d Hz and resampling',
                           self._target_samplerate, default_rate)
            stream = sd.InputStream(samplerate=default_rate, channels=1,
                                    dtype='float32', device=self._device,
                                    callback=self._callback)
            self._stream_samplerate = default_rate
        stream.start()
        self._stream = stream

    def _clear_buffers_locked(self) -> None:
        self._pre_blocks.clear()
        self._pre_samples = 0
        self._rec_blocks = []

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """PortAudio callback. Keep this fast: copy the block and file it."""
        block = indata[:, 0].copy()
        with self._lock:
            if status and status.input_overflow:
                self._overflows += 1
                if self._overflows <= 3:
                    logger.warning('Audio input overflow detected')
            if self._state == RecorderState.RECORDING:
                self._rec_blocks.append(block)
            elif self._state == RecorderState.STANDBY and self._preroll_seconds > 0:
                self._pre_blocks.append(block)
                self._pre_samples += len(block)
                limit = int(self._preroll_seconds * self._stream_samplerate)
                # Trim whole blocks while the buffer still exceeds the pre-roll window
                while self._pre_blocks and \
                        self._pre_samples - len(self._pre_blocks[0]) >= limit:
                    self._pre_samples -= len(self._pre_blocks.popleft())


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler (mono float32). Adequate for speech."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    n_out = int(round(len(audio) * dst_rate / src_rate))
    x_out = np.linspace(0.0, len(audio) - 1, num=n_out)
    return np.interp(x_out, np.arange(len(audio)), audio).astype(np.float32)
