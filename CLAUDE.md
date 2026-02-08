# Bellow - Development Guide

## Project Overview

Bellow is a push-to-talk speech-to-text tool that uses OpenAI Whisper models.
It listens for global hotkeys, captures microphone audio, transcribes via
Whisper, and outputs text via clipboard and/or keyboard emulation.

## Build & Run

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[faster-whisper]" # includes optional faster-whisper backend
pip install -e ".[dev]"            # includes test dependencies
```

Run: `bellow` or `python -m bellow.main`

## Test

```bash
pytest tests/ -v
```

## Project Structure

```
bellow/
  __init__.py
  __version__.py
  main.py              # Entry point, hotkey handling, audio capture
  device_metadata.py   # Audio device enumeration helpers
  soundeffects.py      # Audio feedback (mic on/off/dump sounds)
  transcription.py     # Transcription backend abstraction
  audio/               # .ogg sound effect files
tests/
  test_resample.py
  test_transcription.py
  test_rolling_mic.py
  test_soundeffects.py
```

## Key Architecture

- **RollingMic**: Persistent audio input stream with rolling pre-buffer.
  Keeps a ring buffer of recent audio so the first words aren't lost when the
  hotkey is pressed. Has an idle timeout to release the mic when not in use.
- **TranscriptionBackend**: Abstract interface for speech-to-text. Two
  implementations: `HuggingFaceBackend` (transformers pipeline) and
  `FasterWhisperBackend` (CTranslate2-based, faster inference, lower VRAM).
- **Hotkeys**: pynput GlobalHotKeys for cross-platform key binding.

## Microphone Tension: Design Decision

### Problem

On Linux there is a tension between two failure modes:

1. **Open mic on-demand** (original code): Opening a PortAudio stream takes
   100-300ms. The user starts speaking before the stream is ready, losing the
   first few words. Segments can also be dropped between chunk reads.
2. **Always-open mic** (RollingMic with no timeout): Solves the latency
   problem but the mic device shows as "in use" permanently, blocking other
   apps (Zoom, Teams, system dictation).

### Solution: Idle-Timeout RollingMic

The mic stream stays open during and shortly after recording sessions, then
auto-closes after a configurable idle timeout (`--mic-idle-timeout`, default
30 seconds). This balances the two concerns:

- **During active use**: Mic is open, pre-buffer is warm, no words lost.
- **Between quick dictations**: Mic stays open (within timeout window), so
  rapid successive uses have zero latency.
- **After extended idle**: Mic is released, freeing it for other apps.
- **On next use after timeout**: Mic reopens with ~200ms latency. The pre-roll
  buffer won't have audio from before the reopen, but this is acceptable for
  the "came back after a break" case.

On Linux with PulseAudio/PipeWire (`--host pulse`, the default), multiple apps
can share the mic simultaneously, so the timeout is less critical. On ALSA
(`--host alsa`), exclusive access makes the timeout more important.

### Configuration

```
--mic-idle-timeout 30    # seconds before auto-releasing mic (0 = never release)
--preroll-seconds 0.2    # seconds of pre-hotkey audio to prepend
--prebuffer-seconds 1.0  # rolling buffer size
```

## Whisper Modernization

The original code used the HuggingFace transformers `pipeline()` for Whisper.
This works but is slower and uses more VRAM than alternatives.

### Backend: faster-whisper (recommended)

`faster-whisper` uses CTranslate2 for inference. Benefits:
- 4x faster than HuggingFace pipeline
- ~50% less VRAM
- Supports beam search, VAD filtering
- Drop-in model compatibility (same model names)

Select with `--backend faster-whisper` (falls back to huggingface if not
installed).

### Backend: huggingface (default, no extra deps)

The original transformers pipeline approach. Reliable, well-tested, but slower.

Select with `--backend huggingface`.

## Lint / Style

No linter is currently configured. Follow existing code style (no type stubs,
standard logging, minimal dependencies).
