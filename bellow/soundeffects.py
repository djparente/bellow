"""Audio feedback cues for Bellow.

Effects:
    mic_on   -- recording started (from micon.ogg)
    mic_off  -- recording stopped, transcription queued (from micoff.ogg)
    dump     -- recording discarded / error (from dump.ogg)
    armed    -- microphone armed into standby (generated rising two-tone)
    released -- microphone fully released (generated falling two-tone)

Playback is best-effort: a missing output device or unreadable file logs a
warning but never interrupts recording or transcription.
"""

from __future__ import annotations

import importlib.resources
import logging
import threading
from collections import namedtuple

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

AudioData = namedtuple('AudioData', ['data', 'samplerate'])

_effects: dict[str, AudioData] = {}
_load_lock = threading.Lock()
_loaded = False


def _tone_sequence(frequencies: list[float], tone_seconds: float = 0.09,
                   samplerate: int = 44100, amplitude: float = 0.25) -> AudioData:
    """Generate a sequence of short sine tones with fade-in/out envelopes."""
    segments = []
    n = int(tone_seconds * samplerate)
    fade = int(0.008 * samplerate)
    envelope = np.ones(n, dtype=np.float32)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    for freq in frequencies:
        t = np.arange(n, dtype=np.float32) / samplerate
        segments.append(amplitude * envelope * np.sin(2 * np.pi * freq * t))
    return AudioData(np.concatenate(segments).astype(np.float32), samplerate)


def _read_effect_package(filename: str, effect_name: str) -> None:
    try:
        path = importlib.resources.files('bellow.audio').joinpath(filename)
        _effects[effect_name] = AudioData(*sf.read(str(path)))
    except Exception as e:
        logger.warning('Failed to read audio effect %s from %s: %s', effect_name, filename, e)


def _ensure_loaded() -> None:
    global _loaded
    with _load_lock:
        if _loaded:
            return
        _read_effect_package('micon.ogg', 'mic_on')
        _read_effect_package('micoff.ogg', 'mic_off')
        _read_effect_package('dump.ogg', 'dump')
        _effects['armed'] = _tone_sequence([523.0, 784.0])
        _effects['released'] = _tone_sequence([784.0, 523.0])
        _loaded = True


def play_effect(effect_name: str, blocking: bool = False, device=None) -> None:
    """Play a named feedback sound. Never raises."""
    _ensure_loaded()
    effect_data = _effects.get(effect_name)
    if effect_data is None:
        logger.warning('Unknown audio effect %s', effect_name)
        return
    try:
        sd.play(effect_data.data, effect_data.samplerate, device=device)
        if blocking:
            sd.wait()
    except Exception as e:
        logger.warning('Failed to play audio effect %s: %s', effect_name, e)
