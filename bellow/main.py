"""Bellow: push-to-talk Whisper transcription bound to global hotkeys.

Control flow
------------
Four global hotkeys drive a small state machine (see recorder.py):

    toggle  -- start recording / stop recording and transcribe
    dump    -- stop recording and discard the audio
    mic     -- arm (standby) or fully release the microphone
    quit    -- exit Bellow

While armed, the microphone stream stays open and a short pre-roll ring
buffer is kept, so the first moments of speech are not lost to audio device
start-up latency. The mic hotkey releases the device entirely so other
applications (e.g. video conferencing) can use it.

Transcription runs on a worker thread fed by a queue, so a new recording can
start while the previous one is still being transcribed; results are
delivered in order to the clipboard and/or as emulated keystrokes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from bellow.device_metadata import get_device_name, list_devices
from bellow.hotkeys import HotkeyManager, warn_if_wayland
from bellow.output import OutputSink
from bellow.recorder import Recorder, RecorderState
from bellow.soundeffects import play_effect
from bellow.transcriber import Transcriber

logging.basicConfig(format='%(levelname)s: %(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='bellow',
        description='Global-hotkey speech-to-text using OpenAI Whisper')
    parser.add_argument('-m', '--model', type=str, default='openai/whisper-large-v3-turbo',
                        help='HuggingFace Whisper model to use (default: %(default)s)')
    parser.add_argument('-d', '--device', type=str, default='auto',
                        help="Torch device for inference: 'auto', 'cpu', 'cuda:0', ... "
                             '(default: %(default)s)')
    parser.add_argument('--dtype', type=str, default='auto',
                        choices=['auto', 'float16', 'float32'],
                        help='Inference precision; auto = float16 on CUDA, float32 on CPU')
    parser.add_argument('-l', '--language', type=str, default=None,
                        help='Force the transcription language (default: autodetect)')
    parser.add_argument('--chunk-length', type=float, default=0.0,
                        help='If > 0, use chunked long-form decoding with this chunk size in '
                             'seconds (faster on very long audio, but can lose words at chunk '
                             'boundaries). Default 0 = sequential long-form decoding.')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Audio input device index or name substring; '
                             'see --list-devices (default: system default input)')
    parser.add_argument('--list-devices', action='store_true', default=False,
                        help='Display all audio devices and exit')
    parser.add_argument('--toggle-hotkey', type=str, default='ctrl+shift+alt+f11',
                        help='Hotkey to start recording / stop and transcribe '
                             '(default: %(default)s)')
    parser.add_argument('--dump-hotkey', type=str, default='ctrl+shift+alt+f12',
                        help='Hotkey to stop recording and discard the audio '
                             '(default: %(default)s)')
    parser.add_argument('--mic-hotkey', type=str, default='ctrl+shift+alt+f10',
                        help='Hotkey to arm/release the microphone; releasing frees the '
                             'device for other applications (default: %(default)s)')
    parser.add_argument('--quit-hotkey', type=str, default='ctrl+shift+alt+esc',
                        help='Hotkey to exit bellow (default: %(default)s)')
    parser.add_argument('--preroll', type=float, default=1.0,
                        help='Seconds of standby audio to prepend to each recording so the '
                             'start of speech is not lost (default: %(default)s; 0 disables)')
    parser.add_argument('--no-standby', action='store_true', default=False,
                        help='Do not hold the microphone open between recordings. The device '
                             'stays free for other applications, but pre-roll is unavailable '
                             'and recording start is slower.')
    parser.add_argument('--min-duration', type=float, default=0.25,
                        help='Discard recordings shorter than this many seconds; very short '
                             'clips make Whisper hallucinate (default: %(default)s)')
    parser.add_argument('--no-clipboard', action='store_true', default=False,
                        help='Disable output to the clipboard')
    parser.add_argument('--no-keyboard', action='store_true', default=False,
                        help='Disable output using keyboard emulation')
    return parser


class BellowApp:
    """Owns the recorder, transcriber and output sink, and implements the
    hotkey handlers. Hotkey callbacks are serialized by pynput's listener
    thread, so the handlers below never run concurrently with one another."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.hold_standby = not args.no_standby
        self.quit_event = threading.Event()

        input_device: int | str | None = args.input
        if input_device is not None and input_device.lstrip('-').isdigit():
            input_device = int(input_device)
        self.input_device = input_device

        self.sink = OutputSink(use_clipboard=not args.no_clipboard,
                               use_keyboard=not args.no_keyboard)
        self.recorder = Recorder(device=input_device, samplerate=16000,
                                 preroll_seconds=args.preroll)
        self.transcriber = Transcriber(
            model=args.model, device=args.device, dtype=args.dtype,
            language=args.language, chunk_length_s=args.chunk_length,
            on_result=self.sink.deliver,
            on_error=lambda e: play_effect('dump'),
        )

    # -- hotkey handlers ----------------------------------------------------

    def on_toggle(self) -> None:
        if self.recorder.state == RecorderState.RECORDING:
            audio = self.recorder.stop_recording()
            if not self.hold_standby:
                self.recorder.disable()
            play_effect('mic_off')
            if audio is None or len(audio) < self.args.min_duration * self.recorder.samplerate:
                logger.info('Recording shorter than %.2fs; skipping transcription',
                            self.args.min_duration)
            else:
                self.transcriber.submit(audio)
        else:
            try:
                self.recorder.start_recording()
            except Exception as e:
                logger.error('Could not open the audio input device: %s', e)
                play_effect('dump')
                return
            play_effect('mic_on')

    def on_dump(self) -> None:
        if self.recorder.state == RecorderState.RECORDING:
            self.recorder.cancel_recording()
            if not self.hold_standby:
                self.recorder.disable()
            play_effect('dump')

    def on_mic(self) -> None:
        if self.recorder.state == RecorderState.OFF:
            try:
                self.recorder.enable()
            except Exception as e:
                logger.error('Could not open the audio input device: %s', e)
                play_effect('dump')
                return
            play_effect('armed')
        else:
            discarded = self.recorder.disable()
            play_effect('dump' if discarded else 'released')

    def on_quit(self) -> None:
        logger.info('Quit hotkey pressed')
        self.quit_event.set()

    # -- lifecycle ------------------------------------------------------------

    def run(self) -> int:
        args = self.args

        logger.info('Model: %s', args.model)
        logger.info('Output to clipboard: %s', not args.no_clipboard)
        logger.info('Emulate keyboard presses: %s', not args.no_keyboard)
        logger.info('Input device: %s', get_device_name(self.input_device, kind='input'))
        logger.info('Output device: %s', get_device_name(None, kind='output'))
        logger.info('Toggle hotkey: %s', args.toggle_hotkey)
        logger.info('Dump hotkey: %s', args.dump_hotkey)
        logger.info('Mic arm/release hotkey: %s', args.mic_hotkey)
        logger.info('Quit hotkey: %s', args.quit_hotkey)

        try:
            hotkeys = HotkeyManager({
                args.toggle_hotkey: self.on_toggle,
                args.dump_hotkey: self.on_dump,
                args.mic_hotkey: self.on_mic,
                args.quit_hotkey: self.on_quit,
            })
        except ValueError as e:
            logger.error('Invalid hotkey configuration: %s', e)
            return 1
        except Exception as e:
            logger.error('Could not initialize the global hotkey system: %s', e)
            warn_if_wayland()
            return 1

        self.transcriber.load()
        self.transcriber.start()

        if self.hold_standby:
            try:
                self.recorder.enable()
            except Exception as e:
                logger.error('Could not open the audio input device at startup: %s. '
                             'Use the mic hotkey (%s) to retry, or --list-devices to pick '
                             'another input.', e, args.mic_hotkey)

        try:
            hotkeys.start()
        except Exception as e:
            logger.error('Could not start the global hotkey listener: %s', e)
            warn_if_wayland()
            self.recorder.disable()
            self.transcriber.stop(wait=False)
            return 1

        logger.info('Ready')
        try:
            self.quit_event.wait()
        except KeyboardInterrupt:
            logger.info('Interrupted')

        # Orderly shutdown: stop hotkeys, release the mic, drain the queue
        hotkeys.stop()
        self.recorder.disable()
        self.transcriber.stop(wait=True)
        logger.info('Terminated normally')
        return 0


def main() -> None:
    args = build_parser().parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.no_keyboard and args.no_clipboard:
        logger.error('Both keyboard and clipboard output are disabled. There would be no '
                     'output. Allow at least one of these methods. Aborting.')
        sys.exit(1)

    warn_if_wayland()
    sys.exit(BellowApp(args).run())


if __name__ == '__main__':
    main()
