# Bellow

## Overview

```bellow``` enables a global hotkey to automate use of OpenAI's Whisper model. Push Control+Alt+Shift+F11 to start recording on the default microphone and Control+Alt+Shift+F11 again to stop transcription. Bellow will pass the audio through the Whisper transcription pipeline and (1) emulate the keypresses, and (2) place the transcription on 

Whisper has a 30 second window by default, but Bellow makes use of `transformers`'s chunk_length feature to allow transcription of arbitrary length audio. 

By default, it uses the whisper-medium model. Using this on an nVidia 3080 16GB GPU (Laptop version) it required
about 4.3 GB of video RAM (VRAM) and took about 5 seconds to transcribe 1 minute of spoken audio. Thus, ```bellow``` is suitable for realtime applications.

## Installation

First, create a virtual environment. You will first need to install torch. You will want to install one with CUDA support. You can find the correct installation command using the builder at: https://pytorch.org/get-started/locally/. For example:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then, install the file:

```
pip install bellow
```

## Usage

From within your venv, just type:

```
bellow
```

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

If you want to use a different GPU or 