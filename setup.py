from setuptools import setup, find_packages
from bellow.__version__ import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="bellow",
    version=__version__,
    packages=find_packages(),
    package_data={
        '': ['audio/*.ogg']
    },
    install_requires=[
        "clipboard>=0.0.4",
        "numpy>=1.24.1",
        "pynput>=1.7.6",
        "sounddevice>=0.4.6",
        "soundfile>=0.12.1",
        "transformers>=4.33.2",
    ],
    extras_require={
        "faster-whisper": ["faster-whisper>=0.10.0"],
        "dev": ["pytest>=7.0", "scipy>=1.10.0"],
    },
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
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'bellow=bellow.main:main'
        ]
    }
)
