import soundfile as sf
import sounddevice as sd
from collections import namedtuple
import warnings
import importlib.resources

effects = {}
AudioData = namedtuple('AudioData', ['data', 'samplerate'])

def read_effect(path: str, effect_name: str):
    try:
        effects[effect_name] = AudioData(*sf.read(path))
    except:
        warnings.warn(f'Failed to read audio effect {effect_name} from {path}')

def read_effect_package(filename: str, effect_name: str) -> None:
    path = importlib.resources.files('bellow.audio').joinpath(filename)
    read_effect(str(path), effect_name)

def play_effect(effect_name: str, blocking: bool=False):
    effect_data = effects.get(effect_name, None)

    if not effect_data:
        warnings.warn(f'Failed to play audio effect {effect_name} because it is not in the library of effects')
        return

    sd.play(effect_data.data, effect_data.samplerate)

    if blocking:
        sd.wait()

read_effect_package('micoff.ogg', 'mic_off')
read_effect_package('micon.ogg', 'mic_on')
read_effect_package('dump.ogg', 'dump')