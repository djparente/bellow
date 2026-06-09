"""Whisper transcription worker for Bellow.

Transcription runs on its own worker thread fed by a queue, so a new
recording can begin while the previous one is still being transcribed.
Results are delivered, in order, through the on_result callback.

By default audio is decoded with Whisper's *sequential* long-form algorithm
(the OpenAI reference behaviour), which transformers uses automatically for
inputs longer than 30 s when no chunk_length_s is given. The older chunked
algorithm (chunk_length_s=30) is faster on very long audio but is known to
drop or garble words at chunk boundaries; it remains available via the
--chunk-length option.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    """Resolve a --device argument, mapping 'auto' to cuda if available."""
    if device != 'auto':
        return device
    import torch
    if torch.cuda.is_available():
        return 'cuda:0'
    logger.warning('No CUDA device available; falling back to CPU inference (this may be slow)')
    return 'cpu'


class Transcriber:
    """Loads a Whisper pipeline and transcribes audio on a worker thread."""

    def __init__(self, model: str, device: str = 'auto', dtype: str = 'auto',
                 language: str | None = None, chunk_length_s: float = 0.0,
                 batch_size: int = 8,
                 on_result: Callable[[str], None] | None = None,
                 on_error: Callable[[Exception], None] | None = None):
        """
        :param model: HuggingFace model id, e.g. openai/whisper-large-v3-turbo
        :param device: torch device ('auto', 'cpu', 'cuda:0', ...)
        :param dtype: 'auto' (float16 on cuda, float32 otherwise), 'float16' or 'float32'
        :param language: force a transcription language (None = autodetect)
        :param chunk_length_s: >0 enables the chunked long-form algorithm;
            0 uses sequential long-form decoding (more accurate)
        :param batch_size: batch size for the chunked algorithm
        :param on_result: called from the worker thread with each transcription
        :param on_error: called from the worker thread when transcription fails
        """
        self._model_id = model
        self._device = device
        self._dtype = dtype
        self._language = language
        self._chunk_length_s = float(chunk_length_s)
        self._batch_size = int(batch_size)
        self._on_result = on_result
        self._on_error = on_error

        self._pipe = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def load(self) -> None:
        """Load the model. Heavy imports are kept local so that the rest of
        Bellow can be imported (and tested) without torch installed."""
        import torch
        from transformers import pipeline

        self._device = resolve_device(self._device)
        if self._dtype == 'auto':
            torch_dtype = torch.float16 if self._device.startswith('cuda') else torch.float32
        else:
            torch_dtype = getattr(torch, self._dtype)

        logger.info('Loading model %s on %s (%s)', self._model_id, self._device, torch_dtype)
        self._pipe = pipeline(
            'automatic-speech-recognition',
            model=self._model_id,
            device=self._device,
            torch_dtype=torch_dtype,
        )
        logger.info('Model loaded')

    def start(self) -> None:
        """Start the worker thread (load() must have been called)."""
        if self._pipe is None:
            raise RuntimeError('Transcriber.load() must be called before start()')
        self._thread = threading.Thread(target=self._run, name='transcriber', daemon=True)
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        """Ask the worker to finish queued work and exit."""
        self._queue.put(None)
        if wait and self._thread is not None:
            self._thread.join()

    def submit(self, audio: np.ndarray) -> None:
        """Queue an audio array (float32 mono, 16 kHz) for transcription."""
        self._queue.put(audio)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe synchronously and return the text."""
        kwargs = {}
        if self._chunk_length_s > 0:
            kwargs['chunk_length_s'] = self._chunk_length_s
            kwargs['batch_size'] = self._batch_size
        if self._language:
            kwargs['generate_kwargs'] = {'language': self._language}
        result = self._pipe(audio, **kwargs)
        return result['text'].strip()

    def _run(self) -> None:
        while True:
            audio = self._queue.get()
            if audio is None:
                return
            try:
                text = self.transcribe(audio)
                logger.info('Transcribed %.2fs of audio: %d characters',
                            len(audio) / 16000, len(text))
                if self._on_result is not None:
                    self._on_result(text)
            except Exception as e:
                logger.error('Error during transcription: %s', e)
                if self._on_error is not None:
                    self._on_error(e)
