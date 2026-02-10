import sys
import argparse
import logging
import time
import signal
from collections import deque
import numpy as np
import sounddevice as sd
from pynput import keyboard
import clipboard
from threading import Event, BoundedSemaphore, Lock, Thread, Timer
from typing import Optional, Union

from bellow.device_metadata import list_devices, get_device_name
from bellow.soundeffects import play_effect
from bellow.transcription import (
    TranscriptionBackend, create_backend, WHISPER_SR,
)

# ---------------------------------------
# Logging
# ---------------------------------------
logging.basicConfig(format='%(levelname)s: %(asctime)s - %(message)s', level=logging.INFO)

# ---------------------------------------
# Globals / Shared State
# ---------------------------------------
_backend: TranscriptionBackend | None = None

halt_recording = Event()
halt_recording.set()

pause_recording = Event()
pause_recording.clear()

capture_semaphore = BoundedSemaphore(1)

dump = False
feedback_sound = 'mic_off'
input_device = None

no_keyboard = False
no_clipboard = False

keyboard_controller = keyboard.Controller()

# Serializes clipboard/keyboard output so concurrent transcription threads
# don't interleave their results.
_output_lock = Lock()

# Keep a reference to the hotkey listener so signal handlers can stop it
_hotkeys_listener: keyboard.GlobalHotKeys | None = None

# ---------------------------------------
# Helpers for opening a workable audio stream
# ---------------------------------------
def _device_default_samplerate(device: int | None) -> int | None:
    try:
        info = sd.query_devices(device, 'input')
        dsr = info.get('default_samplerate')
        return int(dsr) if dsr else None
    except Exception:
        return None


def _find_working_samplerate(device: int | None, candidates=(48000, 44100, 32000, 16000)) -> int:
    """
    Return the first samplerate that the device reports as valid, without
    opening a stream.
    """
    ordered: list[int] = []
    default_sr = _device_default_samplerate(device)
    if default_sr:
        ordered.append(default_sr)
    for sr in candidates:
        if sr not in ordered:
            ordered.append(sr)

    for sr in ordered:
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=sr, dtype='int16')
            return sr
        except Exception:
            continue
    raise RuntimeError("No workable input sample rate found for the selected device.")


def _resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    """Resample audio from sr_in to 16 kHz. Returns float32."""
    if sr_in == WHISPER_SR:
        return x.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(WHISPER_SR, sr_in)
        up = WHISPER_SR // g
        down = sr_in // g
        y = resample_poly(x, up, down)
        return y.astype(np.float32, copy=False)
    except Exception:
        # Fallback: linear interpolation
        n_in = x.shape[0]
        if n_in == 0:
            return np.zeros((0,), dtype=np.float32)
        duration = n_in / float(sr_in)
        n_out = int(round(duration * WHISPER_SR))
        if n_out <= 1:
            return np.zeros((0,), dtype=np.float32)
        t_in = np.linspace(0.0, duration, num=n_in, endpoint=False, dtype=np.float64)
        t_out = np.linspace(0.0, duration, num=n_out, endpoint=False, dtype=np.float64)
        y = np.interp(t_out, t_in, x.astype(np.float64))
        return y.astype(np.float32, copy=False)


# ---------------------------------------
# Persistent input stream with rolling pre-buffer and idle timeout
# ---------------------------------------
class RollingMic:
    """
    Persistent input stream with a rolling pre-buffer (int16) and an
    idle-timeout that auto-closes the stream when not in use.

    - Keeps a continuously-updating ring buffer of the last ~prebuf_seconds.
    - When a session starts, seeds frames with the last preroll_s seconds.
    - During a session, keeps appending until stopped.
    - After idle_timeout_s seconds with no active session, the underlying
      PortAudio stream is closed to free the device for other apps.
    - The stream is lazily reopened on the next start_session() call.
    """

    def __init__(self, device: int | None, prebuf_seconds: float = 1.0,
                 idle_timeout_s: float = 30.0):
        self.device = device
        self.prebuf_seconds = max(0.1, float(prebuf_seconds))
        self.idle_timeout_s = float(idle_timeout_s)
        self.lock = Lock()
        self.stream: sd.InputStream | None = None
        self.sr_in: int | None = None
        self.channels = 1
        self.prebuf: deque = deque()
        self.prebuf_samples = 0
        self.prebuf_limit = 0
        self.session_active = False
        self.session_frames: list[np.ndarray] = []
        self._idle_timer: Timer | None = None
        self._closed_permanently = False
        self._open_stream()

    # --- stream lifecycle ---

    def _open_stream(self, retries: int = 2, retry_delay: float = 0.3):
        """
        Open the PortAudio input stream.

        If the device is busy (e.g. exclusive ALSA access held by another app),
        retries a few times before giving up. On failure, self.stream stays None
        and a descriptive error is raised so callers can inform the user.
        """
        if self._closed_permanently:
            return
        if self.stream is not None:
            return  # already open

        self.sr_in = _find_working_samplerate(self.device, candidates=(48000, 44100, 32000, 16000))

        def _mk(ch: int):
            return sd.InputStream(
                samplerate=self.sr_in, channels=ch, dtype='int16',
                device=self.device, latency='low', blocksize=0,
                callback=self._callback,
            )

        last_err = None
        for attempt in range(1 + retries):
            try:
                try:
                    self.stream = _mk(1)
                    self.channels = 1
                except Exception:
                    self.stream = _mk(2)
                    self.channels = 2

                self.prebuf_limit = int(self.sr_in * self.prebuf_seconds)
                self.stream.start()
                logging.info(
                    f'RollingMic opened (SR={self.sr_in} Hz, ch={self.channels}, '
                    f'prebuf≈{self.prebuf_seconds:.2f}s, idle_timeout={self.idle_timeout_s}s)'
                )
                return  # success
            except Exception as e:
                last_err = e
                self.stream = None
                if attempt < retries:
                    logging.warning(
                        f'Mic busy (attempt {attempt + 1}/{1 + retries}): {e} '
                        f'-- retrying in {retry_delay}s'
                    )
                    time.sleep(retry_delay)

        # All retries exhausted
        logging.error(
            f'Could not open microphone after {1 + retries} attempts: {last_err}. '
            f'Another application may have exclusive access to the audio device. '
            f'Close it or switch to --host pulse for shared access.'
        )
        raise RuntimeError(
            f'Microphone unavailable (device busy): {last_err}'
        )

    def _close_stream(self):
        """Close the PortAudio stream (but don't mark permanently closed)."""
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        finally:
            self.stream = None
            self.prebuf.clear()
            self.prebuf_samples = 0
        logging.info('RollingMic stream closed (idle timeout)')

    def _cancel_idle_timer(self):
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _start_idle_timer(self):
        """Schedule auto-close after idle_timeout_s seconds."""
        self._cancel_idle_timer()
        if self.idle_timeout_s <= 0:
            return  # 0 means never auto-close
        self._idle_timer = Timer(self.idle_timeout_s, self._on_idle_timeout)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _on_idle_timeout(self):
        with self.lock:
            if self.session_active:
                # Session started while timer was running; don't close
                return
            self._close_stream()

    @property
    def is_stream_open(self) -> bool:
        return self.stream is not None

    # --- PortAudio callback ---

    def _callback(self, indata, frames, time_info, status):
        chunk = indata.copy()
        with self.lock:
            # Maintain rolling prebuffer
            self.prebuf.append(chunk)
            self.prebuf_samples += frames
            while self.prebuf_samples > self.prebuf_limit and self.prebuf:
                old = self.prebuf.popleft()
                self.prebuf_samples -= old.shape[0]

            # If session running and not paused, append
            if self.session_active and not pause_recording.is_set():
                self.session_frames.append(chunk)

    # --- session API ---

    def start_session(self, preroll_s: float = 0.6):
        """
        Begin a logical capture session; seed with last preroll_s seconds.
        Reopens the stream if it was closed by idle timeout.

        Raises RuntimeError if the microphone cannot be opened (e.g. device
        held exclusively by another application).
        """
        with self.lock:
            self._cancel_idle_timer()
            # Reopen if needed
            if self.stream is None:
                logging.info('RollingMic reopening stream (was idle-closed)')
                self._open_stream()
                # After reopen the prebuffer is empty, so no pre-roll audio
                # will be available -- this is acceptable for the "came back
                # after idle" case.

            preroll_s = max(0.0, float(preroll_s))
            need = int(min(preroll_s, self.prebuf_limit / self.sr_in) * self.sr_in) if self.sr_in else 0
            self.session_frames = []
            if need > 0 and self.prebuf_samples > 0 and self.prebuf:
                buf = np.concatenate(list(self.prebuf), axis=0)
                pre = buf[-need:] if buf.shape[0] >= need else buf
                self.session_frames.append(pre.copy())
            self.session_active = True

    def stop_session(self) -> np.ndarray | None:
        """
        End session and return concatenated int16 ndarray (n, ch), or None.
        Starts the idle timer.
        """
        with self.lock:
            self.session_active = False
            if not self.session_frames:
                result = None
            else:
                result = np.concatenate(self.session_frames, axis=0)
                self.session_frames = []
            self._start_idle_timer()
            return result

    def close(self):
        """Permanently close the stream."""
        self._cancel_idle_timer()
        self._closed_permanently = True
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        finally:
            self.stream = None


# Singleton rolling mic
_rolling_mic: RollingMic | None = None

# ---------------------------------------
# Audio capture via RollingMic
# ---------------------------------------
def audio_capture(device: int | None = None,
                  sample_duration: float = 0.1,
                  preroll_s: float = 0.6) -> np.ndarray | None:
    """
    Persistent-stream capture via RollingMic with pre-roll.
    Waits until halt is requested; the PortAudio callback does the copying.
    """
    global feedback_sound, _rolling_mic

    halt_recording.clear()
    full_sample = None
    try:
        if _rolling_mic is None:
            _rolling_mic = RollingMic(device)

        play_effect('mic_on', blocking=False)
        _rolling_mic.start_session(preroll_s=preroll_s)

        sleep_step = max(0.02, float(sample_duration))
        while not halt_recording.is_set():
            time.sleep(sleep_step)

        play_effect(feedback_sound, blocking=False)
        logging.info('Stopped recording')

        raw = _rolling_mic.stop_session()
        if raw is not None and raw.size > 0:
            # Downmix to mono, normalize to [-1,1], resample to 16k
            if raw.ndim == 2 and raw.shape[1] > 1:
                x = raw.astype(np.float32).mean(axis=1) / 32768.0
            elif raw.ndim == 2 and raw.shape[1] == 1:
                x = raw[:, 0].astype(np.float32) / 32768.0
            else:
                x = raw.astype(np.float32) / 32768.0

            sr_in = _rolling_mic.sr_in if _rolling_mic and _rolling_mic.sr_in else WHISPER_SR
            full_sample = _resample_to_16k(x, sr_in)

    except Exception as e:
        logging.error(f'Error during microphone acquisition: {e}')
        play_effect('dump', blocking=False)
    finally:
        # Guarantee halt_recording is set on every exit path so the toggle
        # never gets stuck thinking a recording is still active.
        halt_recording.set()
    return full_sample


# ---------------------------------------
# Hotkey handlers
# ---------------------------------------
def set_halt(dump_audio: bool = False) -> None:
    global dump, feedback_sound
    dump = dump_audio
    feedback_sound = 'dump' if dump else 'mic_off'
    logging.info('Notifying of halt')
    halt_recording.set()


def run_instance(args=None) -> None:
    global dump, input_device, no_clipboard, no_keyboard, _backend
    did_acquire = capture_semaphore.acquire(blocking=False)
    if not did_acquire:
        logging.info('Semaphore blocked new recording while processing ongoing')
        return

    # --- Recording phase (semaphore held) ---
    try:
        pause_recording.clear()
        captured = audio_capture(
            device=input_device,
            sample_duration=0.1,
            preroll_s=(args.preroll_seconds if args else 0.6),
        )
        # Snapshot dump flag before releasing the semaphore; a new session
        # could overwrite the global between release and the check below.
        should_dump = dump
    finally:
        capture_semaphore.release()

    # --- Transcription & output phase (semaphore released) ---
    # Runs outside the semaphore so the user can immediately start a new
    # recording even while a previous transcription is still processing.
    if captured is not None and not should_dump and _backend is not None:
        try:
            t0 = time.monotonic()
            full_res = _backend.transcribe(
                captured,
                timestamps_mode=args.timestamps if args else "auto",
                no_ts_max_seconds=args.no_ts_max_seconds if args else 29.9,
                task=args.task if args else "auto",
                language=args.language if args else "auto",
            )
            elapsed = time.monotonic() - t0
            if elapsed > 30:
                logging.warning(
                    f'Transcription took {elapsed:.1f}s — consider a smaller '
                    f'model or GPU acceleration'
                )
        except Exception as e:
            logging.error(f'Transcription failed: {e}')
            play_effect('dump', blocking=False)
            return

        if full_res:
            _deliver_output(full_res)


def _deliver_output(text: str) -> None:
    """Deliver transcription result via clipboard and/or keyboard emulation.

    Serialized with _output_lock so concurrent transcription threads don't
    interleave their keyboard output.
    """
    with _output_lock:
        if not no_clipboard:
            try:
                clipboard.copy(text)
            except Exception as e:
                logging.error(f'Clipboard copy failed: {e}')
        if not no_keyboard:
            try:
                for char in text:
                    keyboard_controller.press(char)
                    keyboard_controller.release(char)
                    time.sleep(0.005)
            except Exception as e:
                logging.error(f'Keyboard emulation failed: {e}')


def handle_run_instance(args) -> None:
    Thread(target=run_instance, kwargs={"args": args}, daemon=True).start()


def handle_toggle_factory(args):
    def _inner():
        if halt_recording.is_set():
            handle_run_instance(args)
        else:
            set_halt(dump_audio=False)
    return _inner


def handle_dump() -> None:
    if not halt_recording.is_set():
        set_halt(dump_audio=True)


def handle_pause_toggle() -> None:
    if pause_recording.is_set():
        pause_recording.clear()
        logging.info('Recording resumed (pause cleared)')
        play_effect('reactivate', blocking=False)
    else:
        pause_recording.set()
        logging.info('Recording paused')
        play_effect('pause', blocking=False)


def handle_quit() -> None:
    logging.info("Quit hotkey pressed; shutting down.")
    set_halt(dump_audio=True)
    global _hotkeys_listener, _rolling_mic
    try:
        if _hotkeys_listener is not None:
            _hotkeys_listener.stop()
    except Exception:
        pass
    try:
        if _rolling_mic is not None:
            _rolling_mic.close()
            _rolling_mic = None
    except Exception:
        pass


# ---------------------------------------
# Host API & device selection helpers
# ---------------------------------------
def _pick_hostapi(prefer: Optional[str]) -> None:
    try:
        if prefer is None:
            return
        has = sd.query_hostapis()
        idx = next((i for i, ha in enumerate(has) if prefer.lower() in ha.get('name', '').lower()), None)
        if idx is not None:
            sd.default.hostapi = idx
    except Exception:
        pass


def _resolve_device(spec: Optional[str], want_output: bool) -> Optional[Union[int, str]]:
    """
    spec may be:
      - None: leave as default for the current host API
      - 'default': follow host-API default (return None)
      - integer string: device index
      - name substring (case-insensitive): first matching device
    """
    if spec is None:
        return None
    s = spec.strip()
    if s.lower() == 'default':
        return None
    try:
        return int(s)
    except ValueError:
        pass
    devs = sd.query_devices()
    key = 'max_output_channels' if want_output else 'max_input_channels'
    matches = [i for i, d in enumerate(devs) if d.get(key, 0) > 0 and s.lower() in d.get('name', '').lower()]
    if not matches:
        raise ValueError(f'No {"output" if want_output else "input"} device contains "{spec}".')
    return matches[0]


# ---------------------------------------
# Signal handling
# ---------------------------------------
def _signal_shutdown(signum, frame):
    logging.info(f"Received signal {signum}; shutting down.")
    set_halt(dump_audio=True)
    global _hotkeys_listener, _rolling_mic
    try:
        if _hotkeys_listener is not None:
            _hotkeys_listener.stop()
    except Exception:
        pass
    try:
        if _rolling_mic is not None:
            _rolling_mic.close()
            _rolling_mic = None
    except Exception:
        pass
    try:
        time.sleep(0.2)
    except Exception:
        pass
    sys.exit(0)


# ---------------------------------------
# Main
# ---------------------------------------
def main() -> None:
    global _backend, input_device, no_keyboard, no_clipboard, _hotkeys_listener, _rolling_mic

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default="openai/whisper-medium",
                        help="A HuggingFace OpenAI model to use", dest='model')

    parser.add_argument('--backend', type=str, default="huggingface",
                        choices=["huggingface", "faster-whisper"],
                        help="Transcription backend (default: huggingface)")

    parser.add_argument('--toggle-hotkey', type=str, default="<ctrl>+<shift>+<alt>+<f11>",
                        help='Hotkey sequence to toggle microphone on/off')
    parser.add_argument('--dump-hotkey', type=str, default="<ctrl>+<shift>+<alt>+<f12>",
                        help='Hotkey sequence to toggle microphone off with dumping of audio')
    parser.add_argument('--pause-hotkey', type=str, default="<ctrl>+<shift>+<alt>+<f10>",
                        help='Hotkey sequence to toggle microphone pause during recording')
    parser.add_argument('--quit-hotkey', type=str, default="<ctrl>+<shift>+<alt>+<f9>",
                        help='Hotkey to quit immediately')

    parser.add_argument('-d', '--device', type=str, default="cuda:0",
                        help='The torch device to use (e.g., cuda:0 or cpu)')

    parser.add_argument('-i', '--input', type=int, default=None,
                        help='The index of the input device to use; find the index using --list-devices')
    parser.add_argument('--input-name', type=str, default=None,
                        help='Input device selection by name substring or "default". Overrides --input index.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output device (index or name substring).')

    parser.add_argument('--host', type=str, default='pulse',
                        choices=['pulse', 'alsa'],
                        help='Sound host API to use. "pulse" (default) cooperates with PipeWire/PulseAudio.')

    parser.add_argument('--list-devices', action='store_true', default=False, help="Displays all audio devices")
    parser.add_argument('--no-clipboard', action='store_true', default=False, help="Disables output to the clipboard")
    parser.add_argument('--no-keyboard', action='store_true', default=False,
                        help="Disables output using keyboard emulation")

    parser.add_argument('--task', type=str, default="auto",
                        choices=["auto", "transcribe", "translate"],
                        help="Whisper task: transcribe or translate (to English). 'auto' lets the model decide.")
    parser.add_argument('--language', type=str, default="auto",
                        help="BCP-47 code like 'en', 'es', 'de'. Use 'auto' for language detection.")
    parser.add_argument('--pipe-chunk-length', type=int, default=0,
                        help="Advanced: set pipeline chunk_length_s. 0 disables (recommended).")

    parser.add_argument('--timestamps', type=str, default="auto",
                        choices=["auto", "on", "off"],
                        help="Timestamps mode: auto (enable at >=30s), on (always), off (never).")
    parser.add_argument('--no-ts-max-seconds', type=float, default=29.9,
                        help="When --timestamps=off, chunk length in seconds (default 29.9).")

    parser.add_argument('--preroll-seconds', type=float, default=0.2,
                        help="Seconds of audio captured BEFORE hotkey press to prepend (default 0.2).")
    parser.add_argument('--prebuffer-seconds', type=float, default=1.0,
                        help="Size of the rolling input buffer (default 1.0). Must be >= preroll.")
    parser.add_argument('--mic-idle-timeout', type=float, default=30.0,
                        help="Seconds before auto-releasing mic after recording stops (0 = never release). Default 30.")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    # Choose host API early
    _pick_hostapi(args.host)
    try:
        ha = sd.query_hostapis()[sd.default.hostapi].get('name', '<unknown>')
        logging.info(f'Audio Host API: {ha}')
    except Exception:
        pass

    # Resolve devices
    in_spec = args.input_name if args.input_name is not None else (str(args.input) if args.input is not None else None)
    input_dev = _resolve_device(in_spec, want_output=False)
    output_dev = _resolve_device(args.output, want_output=True)

    try:
        cur_in, cur_out = sd.default.device
    except Exception:
        cur_in, cur_out = (None, None)

    desired_in = input_dev if input_dev is not None else cur_in
    desired_out = output_dev if output_dev is not None else cur_out

    try:
        sd.default.device = (desired_in, desired_out)
    except Exception:
        try:
            if desired_in is not None:
                sd.default.device = (desired_in, sd.default.device[1])
        except Exception:
            pass
        try:
            sd.default.device = (sd.default.device[0], desired_out)
        except Exception:
            pass

    try:
        input_device = sd.default.device[0]
    except Exception:
        input_device = None

    no_keyboard = args.no_keyboard
    no_clipboard = args.no_clipboard

    if no_keyboard and no_clipboard:
        logging.error('Both keyboard and clipboard output is disabled. There will be no output. Aborting.')
        sys.exit(-1)

    logging.info(f'Model: {args.model}')
    logging.info(f'Backend: {args.backend}')
    logging.info(f'Inference Device: {args.device}')
    logging.info(f'Output to clipboard: {not no_clipboard}')
    logging.info(f'Emulate keyboard presses: {not no_keyboard}')
    logging.info(f'Input Device: {get_device_name(idx=sd.default.device[0], input_if_default=True)}')
    logging.info(f'Output Device: {get_device_name(idx=sd.default.device[1], input_if_default=False)}')
    logging.info(f'Mic idle timeout: {args.mic_idle_timeout}s')
    logging.info(f'Toggle Hotkey: {args.toggle_hotkey}')
    logging.info(f'Pause Hotkey: {args.pause_hotkey}')
    logging.info(f'Dump Hotkey: {args.dump_hotkey}')
    logging.info(f'Quit Hotkey: {args.quit_hotkey}')
    logging.info(f'Loading the model {args.model}')

    # Create and load transcription backend
    _backend = create_backend(args.backend)
    _backend.load_model(
        args.model,
        args.device,
        task=args.task,
        language=args.language,
        pipe_chunk_length=args.pipe_chunk_length,
    )

    # Start persistent RollingMic with idle timeout
    try:
        _rolling_mic = RollingMic(
            input_device,
            prebuf_seconds=max(args.prebuffer_seconds, args.preroll_seconds),
            idle_timeout_s=args.mic_idle_timeout,
        )
    except Exception as e:
        logging.warning(f'Failed to start RollingMic early: {e} (will lazy-start on first capture)')

    # Install signal handlers
    try:
        signal.signal(signal.SIGINT, _signal_shutdown)
        signal.signal(signal.SIGTERM, _signal_shutdown)
    except Exception:
        pass

    # Bind hotkeys
    try:
        listener = keyboard.GlobalHotKeys({
            args.toggle_hotkey: handle_toggle_factory(args),
            args.dump_hotkey: handle_dump,
            args.pause_hotkey: handle_pause_toggle,
            args.quit_hotkey: handle_quit,
        })
        _hotkeys_listener = listener
        listener.start()
        logging.info('Ready')
        try:
            listener.join()
        except KeyboardInterrupt:
            _signal_shutdown(signal.SIGINT, None)
    finally:
        _hotkeys_listener = None
        try:
            if _rolling_mic is not None:
                _rolling_mic.close()
        except Exception:
            pass

    logging.info('Terminated normally')
    sys.exit(0)


if __name__ == '__main__':
    main()
