# Bellow

![Bellow Logo: Walruses Bellowing with Bellow on a Blue Background](http://biophysengr.net/files/bellow.png)

## Overview

**Bellow** is a python program that unleashes the power of OpenAI's [Whisper speech-to-text transcription model](https://github.com/openai/whisper).

Specifically, ```bellow``` enables a global hotkey to automate use of OpenAI's Whisper Model. Push Control+Alt+Shift+F11 to start recording on the default microphone and Control+Alt+Shift+F11 again to stop recording. Bellow will pass the audio through the Whisper transcription pipeline and (1) emulate the keypresses, and (2) place the transcription on the clipboard. 

Whisper has a 30-second window by default, but Bellow makes use of `transformers`'s chunk_length feature to allow transcription of arbitrary length audio. 

By default, it uses the whisper-medium model. Using this on an nVidia 3080 16GB GPU (Laptop version) it required
about 4.3 GB of video RAM (VRAM) and took about 5 seconds to transcribe 1 minute of spoken audio. Thus, ```bellow``` is suitable for near-realtime applications. Large models (e.g., whisper-large-v2) may provide better transcription, but will run somewhat slower. On fast GPUs, even the whisper-large-v2 model might be suitable for near-realtime applications.

## Setup

### Requirements
- Bellow runs transcription using OpenAI's Whisper model on a local GPU. This requires a local GPU with enough VRAM to hold the full Whisper model. In principle, you could do inference on the CPU, but this will be very slow and will not likely be suitable for near-realtime applications. I have tested this with NVIDIA GPUs. I am not sure if they would run with AMD using ROCm.
- You will need to be running Python 3.7 or higher (tested with version 3.10.11)
- You will need a system able to run [PyTorch](https://pytorch.org/) on your GPU
- You may need to install the appropriate drivers for GPU

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

You will first need to install torch. You will want to install one with CUDA support. You can find the correct installation command using the builder at: https://pytorch.org/get-started/locally/. For example:
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

When you want to start recording audio push the toggle hotkey (default: ctrl+alt+shift+f11) and dictate for some time. When finished dictating, push the toggle hotkey again and bellow will use Whisper to transcribe your audio to text. It will then simulate keypresses entering this text and also copy it to the clipboard. If, when dictating, you decide you want to stop and discard (not transcribe) the audio, push the dump hotkey instead (default: ctrl+alt+shift+f12). You will receive audio confirmation when the microphone turns on and off.

If you would like to change the hotkeys, you can easily do this by command line argument. You can also change input device, output formats (optionally disable clipboard or keyboard emulation), and can set the device to run inference on.

I mapped the hotkey to a function on my [Razer Tartarus](https://www.razer.com/gaming-keypads/razer-tartarus-v2) (which is why the default hotkey is the way it is -- that combination is otherwise unlikely to be used).

Command line arguments and various options are detailed below.

## Command-line arguments

You can change the model, computing device, and hotkeys using command line arguments:

### Changing the model

`-m` or `--model`: Select a different Whisper Model on HuggingFace, e.g., openai/whisper-large-v2 or openai/whisper-tiny.

These models are described on the HuggingFace model cards: https://huggingface.co/openai/whisper-large-v2 

Examples:

```
bellow --model openai/whisper-large-v2 
```
or
```
bellow --model openai/whisper-tiny 
```

Large models will have better performance but slower inference speed. It seems like the 'medium' model is balances accuracy with real time transcription well.

### Changing the Inference Device

`-d` or `--device` will change the inference device to a different CPU or GPU. By default this uses the GPU with `cuda:0` as the argument. If you want to use a different gpu, you could specify `cuda:1`. Inference on the CPU is supported with `cpu` but is not recommended for real-time applications because it is likely to be very slow.

Example:

```
bellow --device cpu
```

### Change the audio input device (microphone)

`-i` or `--input`: Can change the audio input device by specifying its numerical index. Numerical indices for various devices can be seen by calling:

```
bellow --list-devices
```
Find the device you want in that list (suppose it were to be device 4) and use:
```
bellow --input 4
```

### Change the hotkey

Bellow uses two hotkeys: a toggle hotkey (default ctrl+alt+shift+F11) and a dump hotkey (default ctrl+alt+shift+F12). The toggle hotkey toggles the microphone on/off. When the microphone is toggled off, it uses Whisper to transcribe the audio. If the dump hotkey is used to turn the microphone off instead, it will simply stop recording and disregard the audio.

The hotkeys are configurable using the `--toggle-hotkey` and `--dump-hotkey` arguments, respectively.

Examples:

```
bellow --dump-hotkey "ctrl+shift+d" --toggle-hotkey "ctrl+shift+t"
```

### Disable output modes

By default, bellow will put the resulting transcription on the clipboard and emulate the keypresses. If you want to suppress either of these, you can use the `--no-clipboard` or `--no-keyboard` arguments, respectively. You may not use both simultaneously (because then there is no output).

Example (disables copy to clipboard):

```
bellow --no-clipboard
```

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
