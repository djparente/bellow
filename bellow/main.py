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

# Declare the speech recognition pipeline
pipe = None

# Event for inter-thread communication to stop recording
halt_recording = Event()
halt_recording.set()

# A semaphore to ensure that only one thread is capturing audio data at a time
capture_semaphore = Semaphore()

# A variable to dump the audio instead of using it
dump = False

# A variable that sets the feedback sound to play (when dumping, this is temporarily changed)
feedback_sound = 'mic_off'

input_device = None
no_keyboard = False
no_clipboard = False


def audio_capture(device=None, sample_rate=16000, channels=1, sample_duration=1):
    ''' Capture an infinite amount of audio

    :param device: The sounddevice device to acquire audio on; defaults to the default input
    :param sample_rate: Sample rate of the audio; note that Whisper expects 16000
    :param channels: The number of audio channels; note that Whisper expects 1
    :param sample_duration: The duration of recorded mini-chunks (time between checking for stop)
    :return: A concatenation of the samples as a numpy array ranged from -1 to 1
    '''
    global feedback_sound

    halt_recording.clear()

    samples = []

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype='int16', device=device) as input:
        print('Recording', file=sys.stderr)
        play_effect('mic_on', blocking=False)
        while not halt_recording.is_set():
            sample, overflow = input.read(int(sample_rate * sample_duration))
            samples.append(sample)
        play_effect(feedback_sound, blocking=False)
        print("Stopped recording", file=sys.stderr)

    full_sample = None
    if len(samples) > 0:
        full_sample = np.concatenate(samples).flatten().astype(np.float32) / 32768.0

    return full_sample


def run_whisper(audio_sample):
    '''Runs Whisper on the audio sample and extracts the text
    :param audio_sample: A np.float32 array of mono audio with sample rate 16000
0    :return: The transcribed text by whisper
    '''
    result = pipe(audio_sample.copy(), batch_size=8)["text"].strip()
    return result


def set_halt(dump_audio: bool = False) -> None:
    '''Handler method for a keypress event to stop recording

    :param dump_audio: Whether to ignore this audio

    :return: None
    '''
    global dump, feedback_sound
    dump = dump_audio

    if dump:
        feedback_sound = 'dump'
    else:
        feedback_sound = 'mic_off'

    print('Notifying of halt', file=sys.stderr)
    halt_recording.set()


def run_instance() -> None:
    '''Runs an audio capture and transcribes it.

    :return: None
    '''
    global dump, input_device, no_clipboard, no_keyboard

    did_acquire_semaphore = capture_semaphore.acquire(blocking=False)

    if not did_acquire_semaphore:
        print('Semaphore blocked new recording while processing ongoing', file=sys.stderr)
        return

    captured = audio_capture(device=input_device)

    if captured is not None and not dump:
        full_res = run_whisper(captured)

        # Write the data to clipboard and screen
        if not no_clipboard:
            clipboard.copy(full_res)
        if not no_keyboard:
            keyboard.write(full_res)

    # Release the semaphore
    capture_semaphore.release()


def handle_run_instance() -> None:
    ''' Handler method to start an instance (which will succeed if one is not already running)

    :return: None
    '''
    thread = threading.Thread(target=run_instance)
    thread.start()


def handle_toggle() -> None:
    ''' Handler method to toggle recording on/off

    :return: None
    '''
    global dump

    if halt_recording.is_set():
        handle_run_instance()
    else:
        set_halt(dump_audio=False)


def handle_dump() -> None:
    '''
    Dumps the audio currently being recorded, if any is being recorded

    :return:
    '''
    global dump

    if not halt_recording.is_set():
        set_halt(dump_audio=True)


def list_devices():
    devices = sd.query_devices()
    print(devices)


def main():
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
        print('Both keyboard and clipboard output is disabled. There will be no output. Allow one of these methods. '
              'Aborting.',
              file=sys.stderr)
        sys.exit(-1)

    print(f'Loading the model {args.model}...', file=sys.stderr)

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
    print('Ready', file=sys.stderr)

    # Wait for keyboard hotkeys until Ctrl+Shift+Alt+Esc is detected
    keyboard.wait('ctrl+shift+alt+esc')

    # Notify of exit
    print("Exited main", file=sys.stderr)


if __name__ == '__main__':
    main()
