# Bellow — System Design

Bellow is a push-to-talk dictation tool: a global hotkey records microphone
audio, OpenAI's Whisper transcribes it locally, and the text is delivered to
the clipboard and/or as emulated keystrokes.

This document describes the architecture as of version 2.0, the problems in
the original design that motivated it, and the trade-offs taken.

## Goals

1. **No lost speech.** Neither at the start of a recording (device start-up
   latency) nor in the middle (buffer overflows, chunked-decoding artifacts).
2. **The microphone must be releasable.** Holding a stream open gives instant
   starts, but the user must be able to free the device for other
   applications (e.g. Zoom) with a single hotkey.
3. **Linux first (X11), Windows supported.** No root privileges required.
   Wayland is explicitly out of scope for global hotkeys (see
   "Platform notes").
4. **Robustness.** Errors in any one component (audio device, clipboard,
   model) degrade gracefully with audible + logged feedback; they never
   crash the program or wedge its state machine.

## Module overview

```
bellow/
├── main.py            CLI, BellowApp: wires hotkeys → recorder → transcriber → output
├── recorder.py        Recorder: persistent input stream + pre-roll ring buffer state machine
├── transcriber.py     Transcriber: Whisper pipeline on a worker thread fed by a queue
├── hotkeys.py         HotkeyManager: pynput global hotkeys + friendly spec parsing
├── output.py          OutputSink: clipboard (pyperclip) and keystroke (pynput) delivery
├── soundeffects.py    Audible feedback cues (ogg files + generated tones)
├── device_metadata.py Audio device listing/naming helpers
└── audio/             micon.ogg, micoff.ogg, dump.ogg
```

## The recorder state machine

The heart of the redesign is `Recorder`, a three-state machine around a
single callback-driven `sounddevice.InputStream`:

```
            mic hotkey (arm)                 toggle hotkey
   ┌─────┐ ───────────────────► ┌─────────┐ ───────────────► ┌───────────┐
   │ OFF │                      │ STANDBY │                  │ RECORDING │
   └─────┘ ◄─────────────────── └─────────┘ ◄─────────────── └───────────┘
            mic hotkey (release)             toggle (stop+transcribe)
                                             dump  (stop+discard)
```

- **OFF** — no stream exists. The microphone is fully released and available
  to any other application. The *mic hotkey* (default `ctrl+shift+alt+f10`)
  toggles between OFF and STANDBY; pressing it during a recording discards
  the recording and releases the device immediately.
- **STANDBY** — the stream is open and the PortAudio callback maintains a
  ring buffer of the most recent `--preroll` seconds (default 1.0 s) of
  audio. Nothing is stored beyond that window.
- **RECORDING** — the callback appends every incoming block to the active
  recording. On entry, the current contents of the ring buffer are moved to
  the front of the recording, so speech that began *before* the hotkey was
  fully pressed — or during what used to be stream start-up time — is
  captured.

Bellow starts in STANDBY by default. `--no-standby` inverts the policy: the
stream is opened only for the duration of each recording and closed
afterwards, keeping the device free at all times at the cost of pre-roll and
start-up latency. Pressing the toggle hotkey while OFF always works: the
stream is opened on demand (without pre-roll, since none could have been
buffered).

### Why a callback instead of blocking reads

The original implementation looped on `stream.read(1 second)`:

- If the Python control thread was delayed (GC, GIL contention, model
  loading), PortAudio's internal buffer overflowed and **samples were
  silently dropped** — the `overflow` flag returned by `read()` was ignored.
  This is the most likely cause of the "lost segments in the middle"
  symptom.
- Stopping had up to one second of latency (the loop only checked the stop
  flag between whole-second reads), appending unwanted trailing audio.

In the new design PortAudio invokes `Recorder._callback` from its own
high-priority thread the moment each block is ready. The callback only
copies the block and appends it to a Python list under a briefly-held lock —
no allocation-heavy work, no inference, no I/O. Overflows reported in the
callback `status` are counted and surfaced as a warning when the recording
ends, so dropped audio is at least *visible* instead of silent.

Locking note: `stream.close()` waits for an in-flight callback to return,
and the callback takes the recorder lock — so the stream is always closed
*outside* the lock to avoid a deadlock (`disable()` swaps the stream handle
out under the lock, then closes it after releasing).

### Sample-rate handling

Whisper requires 16 kHz mono. The recorder asks the device for 16 kHz
directly (capturing `float32`, which Whisper consumes natively — no int16
round-trip). If the device refuses that rate (common with some Windows
WASAPI configurations), the recorder falls back to the device's default
rate and linearly resamples to 16 kHz when the recording is finalized.
Linear interpolation is adequate for speech feeding a model that
mel-filters the signal anyway, and avoids a scipy/librosa dependency.

## Transcription pipeline

`Transcriber` owns the HuggingFace `automatic-speech-recognition` pipeline
and a worker thread fed by a `queue.Queue`:

- **Capture and inference are decoupled.** The old design held a semaphore
  across capture *and* transcription, so the toggle hotkey was silently
  ignored while Whisper was still working. Now a new recording can begin
  immediately; results are still delivered in submission order because there
  is exactly one worker.
- **Sequential long-form decoding by default.** The old code forced
  `chunk_length_s=30`, i.e. transformers' *chunked* algorithm, which is
  known to drop or duplicate words at chunk boundaries — another plausible
  source of mid-speech text loss even when the audio was fine. Modern
  transformers (≥ 4.45) automatically uses Whisper's *sequential* long-form
  algorithm (the OpenAI reference behaviour) for inputs over 30 s when no
  chunk length is given. `--chunk-length N` restores chunked+batched
  decoding for users who prefer throughput on very long recordings.
- **Default model `openai/whisper-large-v3-turbo`** — comparable parameter
  count to the previous default (`whisper-medium`) but markedly more
  accurate and faster at inference.
- **Precision**: `--dtype auto` selects float16 on CUDA (roughly halving
  VRAM versus the old float32-only behaviour) and float32 on CPU.
- **Device**: `--device auto` picks `cuda:0` when available and falls back
  to CPU with a warning, instead of crashing on GPU-less machines.
- **Hallucination guard**: recordings shorter than `--min-duration`
  (default 0.25 s, e.g. an accidental double-tap) are skipped, because
  Whisper reliably hallucinates text ("Thank you.") on near-empty input.
- A failed transcription logs the error, plays the error cue, and leaves the
  worker alive for the next job (the old code raised `UnboundLocalError`
  from its own error handler).

## Hotkeys

`hotkeys.py` uses **pynput** rather than the previous `keyboard` package:

| | `keyboard` (old) | `pynput` (new) |
|---|---|---|
| Linux backend | `/dev/input` — **requires root** | X11 XRecord — no privileges |
| Windows | Win32 hooks | Win32 hooks |
| Wayland | no (reads evdev, but can't inject per-window) | no |

Users keep writing hotkeys in the familiar `ctrl+shift+alt+f11` style;
`to_pynput_spec()` converts to pynput's `<ctrl>+<shift>+<alt>+<f11>` syntax
(specs already containing `<` pass through). All four bindings (toggle,
dump, mic, quit) are validated and checked for duplicates at startup, so a
typo fails immediately rather than at first keypress.

pynput's `GlobalHotKeys` dispatches all callbacks sequentially on its
listener thread. This is a deliberate simplification: the hotkey handlers in
`BellowApp` can never race each other, which eliminates the old design's
toggle race (a fast double-press could observe stale `halt_recording` state
and try to start a second capture instead of stopping the first). The only
remaining concurrency is between the handlers and the audio callback, which
the recorder's internal lock covers. Handler exceptions are caught and
logged so a transient device error cannot kill the listener thread.

## Output

`OutputSink` delivers each transcription to the clipboard (`pyperclip` —
which is what the old `clipboard` package wrapped anyway; on Linux it needs
`xclip` or `xsel`) and/or as typed keystrokes (`pynput.keyboard.Controller`,
XTest on X11 / SendInput on Windows). Each output is attempted
independently; a clipboard failure does not prevent typing and vice versa.

## Audible feedback

Five cues, all best-effort (a missing output device logs a warning, never
raises):

| cue | sound | meaning |
|---|---|---|
| `mic_on` | micon.ogg | recording started |
| `mic_off` | micoff.ogg | recording stopped, transcription queued |
| `dump` | dump.ogg | recording discarded, or any error |
| `armed` | generated rising two-tone | microphone armed into standby |
| `released` | generated falling two-tone | microphone fully released |

The standby cues are synthesized sine sweeps (numpy) so no new binary assets
were needed, and they are audibly distinct from the recording cues.

## Threading model summary

| Thread | Owner | Work |
|---|---|---|
| main | `BellowApp.run` | startup, then waits on the quit event |
| pynput listener | `HotkeyManager` | hotkey detection, all state transitions |
| PortAudio callback | `Recorder` | copy audio blocks into buffers (lock held briefly) |
| transcriber worker | `Transcriber` | Whisper inference, then output delivery |
| sounddevice playback | `soundeffects` | feedback cues |

Shared state is confined to the recorder's lock-protected buffers and the
thread-safe transcription queue; there are no module-level mutable globals
(the old `dump` / `feedback_sound` / `halt_recording` globals are gone).

## Platform notes

- **Linux / X11**: fully supported, no root required. Requires
  `libportaudio2` and `xclip` or `xsel` for the clipboard.
- **Linux / Wayland**: global hotkeys cannot be observed by an unprivileged
  process by design of the protocol; portal-based shortcuts
  (`GlobalShortcuts` portal) vary by compositor and would have significantly
  complicated the design. Per the project's priorities, Wayland is not
  targeted; Bellow detects `XDG_SESSION_TYPE=wayland` and prints a clear
  warning at startup.
- **Windows**: pynput, sounddevice and pyperclip all work natively; the
  PortAudio binary ships inside the sounddevice wheel.

## Shutdown

The quit hotkey (default `ctrl+shift+alt+esc`) or Ctrl-C sets an event the
main thread is waiting on; main then stops the hotkey listener, releases the
microphone, and joins the transcriber worker so any queued transcription is
finished (and delivered) before exit.

## Testing

`tests/` contains 25 unit tests that run without audio hardware or a GPU:
the PortAudio stream is replaced with a fake and audio blocks are injected
straight into the recorder callback. Covered: every recorder state
transition, ring-buffer trimming, pre-roll inclusion, overflow accounting,
resampling, hotkey spec parsing, and the `BellowApp` handler wiring
(including the min-duration guard and mic-release-during-recording).

```
python -m unittest discover -s tests
```
