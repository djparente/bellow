import argparse
import sys
import numpy as np
import threading
from threading import Event, Semaphore
import sounddevice as sd
import keyboard
import clipboard
from bellow.soundeffects import play_effect
from transformers import pipeline
from typing import Optional
import logging

# Set up logging
logging.basicConfig(format='%(levelname)s: %(asctime)s - %(message)s', level=logging.INFO)

# Declare the speech recognition pipeline
pipe = None
pipelock = threading.Lock()

# Event for inter-thread communication to stop recording
halt_recording = Event()
halt_recording.set()

# A semaphore to ensure that only one thread is capturing audio data at a time
capture_semaphore = Semaphore()

# A variable to dump the audio instead of using it
dump = False

# A variable that sets the feedback sound to play (when dumping, this is temporarily changed)
feedback_sound = 'mic_off'

# A variable that keeps track of the input microphone
input_device = None

# Status variables for how to return results
no_keyboard = False
no_clipboard = False


def audio_capture(device: int | None = None, sample_rate: int = 16000, channels: int = 1,
                  sample_duration: int = 1) -> np.ndarray[np.float32] | None:
    """ Capture an infinite amount of audio

    :param device: The sounddevice device to acquire audio on; defaults to the default input
    :param sample_rate: Sample rate of the audio; note that Whisper expects 16000
    :param channels: The number of audio channels; note that Whisper expects 1
    :param sample_duration: The duration of recorded mini-chunks (time between checking for stop)
    :return: A concatenation of the samples as a numpy array ranged from -1 to 1
    """
    global feedback_sound

    halt_recording.clear()

    samples = []
    full_sample = None

    try:
        with sd.InputStream(samplerate=sample_rate, channels=channels, dtype='int16', device=device) as input_stream:
            logging.info('Recording')
            play_effect('mic_on', blocking=False)
            while not halt_recording.is_set():
                sample, overflow = input_stream.read(int(sample_rate * sample_duration))
                samples.append(sample)
            play_effect(feedback_sound, blocking=False)
            logging.info('Stopped recording')

        if len(samples) > 0:
            full_sample = np.concatenate(samples).flatten().astype(np.float32) / 32768.0

    except Exception as e:
        logging.error(f'Error during microphone acquisition: {e}')
        halt_recording.set()
        play_effect('dump', blocking=False)

    return full_sample


def run_whisper(audio_sample: np.ndarray[np.float32]) -> str:
    """Runs Whisper on the audio sample and extracts the text
    :param audio_sample: A np.float32 array of mono audio with sample rate 16000
0    :return: The transcribed text by whisper
    """
    with pipelock:
        try:
            result = pipe(audio_sample.copy(), batch_size=8)["text"].strip()
        except Exception as e:
            logging.error('Error in transcription {e}')
            play_effect('dump', blocking=False)

    return result


def set_halt(dump_audio: bool = False) -> None:
    """Handler method for a keypress event to stop recording

    :param dump_audio: Whether to ignore this audio

    :return: None
    """
    global dump, feedback_sound
    dump = dump_audio

    if dump:
        feedback_sound = 'dump'
    else:
        feedback_sound = 'mic_off'

    logging.info('Notifying of halt')
    halt_recording.set()


def run_instance() -> None:
    """Runs an audio capture and transcribes it.

    :return: None
    """
    global dump, input_device, no_clipboard, no_keyboard

    did_acquire_semaphore = capture_semaphore.acquire(blocking=False)

    if not did_acquire_semaphore:
        logging.info('Semaphore blocked new recording while processing ongoing')
        return

    try:
        captured = audio_capture(device=input_device)

        if captured is not None and not dump:
            full_res = run_whisper(captured)

            # Write the data to clipboard and screen
            if not no_clipboard:
                clipboard.copy(full_res)
            if not no_keyboard:
                keyboard.write(full_res)
    finally:
        # Release the semaphore
        capture_semaphore.release()


def handle_run_instance() -> None:
    """ Handler method to start an instance (which will succeed if one is not already running)

    :return: None
    """
    thread = threading.Thread(target=run_instance)
    thread.start()


def handle_toggle() -> None:
    """ Handler method to toggle recording on/off

    :return: None
    """
    global dump

    if halt_recording.is_set():
        handle_run_instance()
    else:
        set_halt(dump_audio=False)


def handle_dump() -> None:
    """
    Dumps the audio currently being recorded, if any is being recorded

    :return:
    """
    global dump

    if not halt_recording.is_set():
        set_halt(dump_audio=True)


def list_devices() -> None:
    """
    Prints a list of all input and output audio devices

    :return: None
    """
    devices = sd.query_devices()
    print(devices)


def get_device_name(idx: Optional[int] = None, input_if_default: Optional[bool] = None) -> str:
    """
    Returns the name of a device with index idx. If this is none, it returns the name of the
    default input or output device depending on the value of input_if_default. Both inputs cannot
    be simultaneously unspecified (None).

    :param idx: The index of the device in the query_devices list
    :param input_if_default: If idx is None, should we return the default input device (if false, return output device)
    :return: The name of the specified device
    """
    # If index is none, get one of the default devices
    if idx is None:
        if input_if_default is None:
            raise ValueError('Cannot retrieve input/output device if input_if_default is None')

        if input_if_default:
            return sd.query_devices(sd.default.device[0])['name']
        else:
            return sd.query_devices(sd.default.device[1])['name']

    # Otherwise return name by index
    devices = sd.query_devices()
    return devices[idx]['name']


def main() -> None:
    """
    Runs the program
    :return: None
    """
    global pipe, input_device, no_keyboard, no_clipboard

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="openai/whisper-medium",
                        help="A HuggingFace OpenAI model to use", dest='model')
    parser.add_argument('--toggle-hotkey', type=str, default="ctrl+shift+alt+f11",
                        help='Hotkey sequence to toggle microphone on/off')
    parser.add_argument('--dump-hotkey', type=str, default="ctrl+shift+alt+f12",
                        help='Hotkey sequence to toggle microphone off with dumping of audio')
    parser.add_argument('-d', '--device', type=str, default="cuda:0",
                        help='The torch device to use (e.g., cuda:0 or cpu)')
    parser.add_argument('-i', '--input', type=int, default=None,
                        help='The index of the input device to use; find the index using --list-device')
    parser.add_argument('--list-devices', action='store_true', default=False, help="Displays all audio devices")
    parser.add_argument('--no-clipboard', action='store_true', default=False, help="Disables output to the clipboard")
    parser.add_argument('--no-keyboard', action='store_true', default=False,
                        help="Disables output using keyboard emulation")

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.input is not None:
        input_device = int(args.input)

    no_keyboard = args.no_keyboard
    no_clipboard = args.no_clipboard

    if no_keyboard and no_clipboard:
        logging.error('Both keyboard and clipboard output is disabled. There will be no output. Allow one of these '
                      'methods. Aborting.')
        sys.exit(-1)

    # Some logging
    logging.info(f'Model: {args.model}')
    logging.info(f'Inference Device: {args.device}')
    logging.info(f'Output to clipboard: {not no_clipboard}')
    logging.info(f'Emulate keyboard presses: {not no_keyboard}')
    logging.info(f'Input Device: {get_device_name(idx=input_device, input_if_default=True)}')
    logging.info(f'Output Device: {get_device_name(idx=None, input_if_default=False)}')
    logging.info(f'Toggle Hotkey: {args.toggle_hotkey}')
    logging.info(f'Dump Hotkey: {args.dump_hotkey}')
    logging.info(f'Loading the model {args.model}')

    with pipelock:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model,
            chunk_length_s=30,
            device=args.device
        )

    # Register a hotkey
    keyboard.add_hotkey(args.toggle_hotkey, handle_toggle)
    keyboard.add_hotkey(args.dump_hotkey, handle_dump)

    # Notify load is completed
    logging.info('Ready')

    # Wait for keyboard hotkeys until Ctrl+Shift+Alt+Esc is detected
    keyboard.wait('ctrl+shift+alt+esc')

    keyboard.remove_hotkey(args.toggle_hotkey)
    keyboard.remove_hotkey(args.dump_hotkey)

    # Notify of exit
    logging.info('Terminated normally')

    exit(0)


if __name__ == '__main__':
    main()
