# Bellow

![Bellow Logo: Walruses Bellowing with Bellow on a Blue Background](http://biophysengr.net/files/bellow.png)

## Overview

**Bellow** is a python program that unleashes the power of OpenAI's [Whisper speech-to-text transcription model](https://github.com/openai/whisper).

Specifically, ```bellow``` enables global hotkeys to automate use of OpenAI's Whisper model. Push Control+Alt+Shift+F11 to start recording on the microphone and Control+Alt+Shift+F11 again to stop. Bellow will pass the audio through the Whisper transcription pipeline and (1) emulate the keypresses, and (2) place the transcription on the clipboard.

To make sure the very first moments of your speech are never lost to audio-device start-up latency, Bellow keeps the microphone in a low-cost *standby* mode between recordings and prepends a short pre-roll buffer (default: 1 second) to each recording. When you need the microphone for something else (a Zoom call, for example), press the *mic hotkey* (default Control+Alt+Shift+F10) to fully release the device; press it again to re-arm.

By default, Bellow uses the `openai/whisper-large-v3-turbo` model with float16 inference on the GPU, which is both faster and more accurate than older medium-size models while using a similar amount of VRAM. Transcription of arbitrary-length audio uses Whisper's sequential long-form decoding for best accuracy.

## Hotkeys

| Hotkey (default) | Action |
|---|---|
| `ctrl+alt+shift+f11` | Start recording / stop and transcribe |
| `ctrl+alt+shift+f12` | Stop recording and discard the audio |
| `ctrl+alt+shift+f10` | Arm/release the microphone (release frees the device for other apps) |
| `ctrl+alt+shift+esc` | Quit Bellow |

You will hear distinct audio cues when recording starts, stops, is discarded, and when the microphone is armed or released.

## Setup

### Requirements
- Bellow runs transcription using OpenAI's Whisper model locally. A CUDA GPU is strongly recommended for near-realtime use; CPU inference works (`--device cpu` or automatically when no GPU is found) but is slow.
- Python 3.10 or higher.
- A system able to run [PyTorch](https://pytorch.org/) (install torch separately, see below).
- **Linux**: an X11 session (global hotkeys do not work under Wayland), the PortAudio library (`sudo apt install libportaudio2`), and `xclip` or `xsel` for clipboard support (`sudo apt install xclip`). Root privileges are **not** required.
- **Windows**: no extra system packages needed.

### Installation

#### Create a virtual environment (venv)
First, create a virtual environment and activate it.

Creation of the virtual env
```
python -m venv venv
```

Activation in linux:
```
source venv/bin/activate
```

Activation in Windows (Powershell):
```
.\venv\Scripts\activate.ps1
```

Activation in Windows (cmd.exe):
```
.\venv\Scripts\activate.bat
```

#### Install Torch

You will first need to install torch, ideally with CUDA support. You can find the correct installation command using the builder at: https://pytorch.org/get-started/locally/. For example:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Install Bellow

```
pip install bellow
```

## Usage

From within your venv, just type:

```
bellow
```

When you want to start recording audio push the toggle hotkey (default: ctrl+alt+shift+f11) and dictate for some time. When finished dictating, push the toggle hotkey again and bellow will use Whisper to transcribe your audio to text. It will then simulate keypresses entering this text and also copy it to the clipboard. If, when dictating, you decide you want to stop and discard (not transcribe) the audio, push the dump hotkey instead (default: ctrl+alt+shift+f12).

Between recordings Bellow keeps the microphone open in standby so recording starts instantly and includes a one-second pre-roll. If you need the microphone for another application, press the mic hotkey (default: ctrl+alt+shift+f10) to release the device; press it again when you want Bellow to take the microphone back. If you prefer Bellow to never hold the microphone between recordings, run it with `--no-standby` (at the cost of pre-roll and a slower recording start).

I mapped the hotkey to a function on my [Razer Tartarus](https://www.razer.com/gaming-keypads/razer-tartarus-v2) (which is why the default hotkey is the way it is -- that combination is otherwise unlikely to be used).

Command line arguments and various options are detailed below.

## Command-line arguments

### Changing the model

`-m` or `--model`: Select a different Whisper model on HuggingFace, e.g., `openai/whisper-large-v3` or `openai/whisper-tiny`. Default: `openai/whisper-large-v3-turbo`.

These models are described on the HuggingFace model cards: https://huggingface.co/openai/whisper-large-v3-turbo

Examples:

```
bellow --model openai/whisper-large-v3
```
or
```
bellow --model openai/whisper-tiny
```

### Changing the Inference Device and precision

`-d` or `--device` selects the inference device. The default, `auto`, uses `cuda:0` when a CUDA GPU is available and falls back to the CPU otherwise. You may specify a device explicitly, e.g. `cuda:1` or `cpu`.

`--dtype` selects the inference precision: `auto` (default; float16 on CUDA, float32 on CPU), `float16`, or `float32`.

Example:

```
bellow --device cpu
```

### Change the audio input device (microphone)

`-i` or `--input`: Select the audio input device by numerical index or by a substring of its name. Indices and names can be seen by calling:

```
bellow --list-devices
```
Find the device you want in that list (suppose it were to be device 4) and use:
```
bellow --input 4
```

### Change the hotkeys

The four hotkeys are configurable using `--toggle-hotkey`, `--dump-hotkey`, `--mic-hotkey` and `--quit-hotkey`. Hotkeys are written in the familiar `modifier+modifier+key` style.

Example:

```
bellow --dump-hotkey "ctrl+shift+d" --toggle-hotkey "ctrl+shift+t"
```

### Recording behaviour

- `--preroll SECONDS`: how much standby audio to prepend to each recording (default 1.0; 0 disables).
- `--no-standby`: never hold the microphone open between recordings. The device stays free for other applications, but pre-roll is unavailable and recording start is slower.
- `--min-duration SECONDS`: discard recordings shorter than this (default 0.25). Very short clips make Whisper hallucinate text.

### Transcription behaviour

- `-l` / `--language`: force the transcription language (default: autodetect), e.g. `--language en`.
- `--chunk-length SECONDS`: if greater than 0, use transformers' chunked long-form decoding with this chunk size. This is faster on very long recordings but can lose or garble words at chunk boundaries. The default (0) uses Whisper's sequential long-form algorithm, which is more accurate.

### Disable output modes

By default, bellow will put the resulting transcription on the clipboard and emulate the keypresses. If you want to suppress either of these, you can use the `--no-clipboard` or `--no-keyboard` arguments, respectively. You may not use both simultaneously (because then there is no output).

Example (disables copy to clipboard):

```
bellow --no-clipboard
```

## Design

See [Design.md](Design.md) for a full description of the architecture, including the recorder state machine, pre-roll ring buffer, and threading model.

## Warnings and other disclaimers

- I am not affiliated with OpenAI in any way.
- Generative artificial intelligence is an advanced tool that is incompletely understood. It may harbor biases, produce unacceptable or offensive content, or provide inaccurate transcriptions.
- I strongly recommend against using this software in any setting may expose humans to harm due to transcription errors (e.g., medical dictation, military applications, etc.)
- The logo for Bellow is made by Generative AI using [Stable Diffusion](https://github.com/CompVis/stable-diffusion). I did perform some searching to make sure the AI was not blatantly copying any existing artwork.
- Use at your own risk.

## Contact

Please feel free to contact me or open an issue with questions or concerns!

## Citation

I have no current plans to submit this anywhere for publication. I would appreciate an acknowledgement if you find this software useful for your applications. Should you want to formally cite this in an academic publication, please cite the repository URL.
