from setuptools import setup, find_packages
from bellow.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bellow",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "certifi==2022.12.7"
        "cffi==1.15.1"
        "charset-normalizer==2.1.1"
        "clipboard==0.0.4"
        "colorama==0.4.6"
        "filelock==3.9.0"
        "fsspec==2023.9.1"
        "huggingface-hub==0.17.2"
        "idna==3.4"
        "Jinja2==3.1.2"
        "keyboard==0.13.5"
        "MarkupSafe==2.1.2"
        "mpmath==1.2.1"
        "networkx==3.0"
        "numpy==1.24.1"
        "packaging==23.1"
        "Pillow==9.3.0"
        "pycparser==2.21"
        "pyperclip==1.8.2"
        "PyYAML==6.0.1"
        "regex==2023.8.8"
        "requests==2.28.1"
        "safetensors==0.3.3"
        "sounddevice==0.4.6"
        "soundfile==0.12.1"
        "sympy==1.11.1"
        "tokenizers==0.13.3"
        "tqdm==4.66.1"
        "transformers==4.33.2"
        "typing_extensions==4.4.0"
        "urllib3==1.26.13"
    ],
    author="Daniel J. Parente",
    author_email="dan.parente@gmail.com",
    description="Implements a pushbutton interface to the Whisper transformer using a global hotkey",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djparente/bellow",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'bellow=bellow.main:main'
        ]
    }
)