"""
Transcription backend abstraction for Bellow.

Provides a common interface for speech-to-text engines:
- HuggingFaceBackend: uses transformers pipeline (original approach)
- FasterWhisperBackend: uses faster-whisper / CTranslate2 (faster, less VRAM)
"""

import logging
from abc import ABC, abstractmethod
from threading import Lock
from typing import Optional

import numpy as np

from bellow.soundeffects import play_effect

# Whisper expects 16 kHz mono float32 in [-1, 1]
WHISPER_SR = 16000


def _seconds_of(audio_16k: np.ndarray) -> float:
    return float(audio_16k.shape[0]) / float(WHISPER_SR) if audio_16k is not None and audio_16k.size > 0 else 0.0


def _split_audio(x: np.ndarray, max_seconds: float) -> list:
    """Split audio into chunks of at most max_seconds."""
    if max_seconds <= 0:
        return [x]
    n = x.shape[0]
    step = int(round(max_seconds * WHISPER_SR))
    if step <= 0 or n == 0:
        return [x]
    return [x[i:min(i + step, n)] for i in range(0, n, step)]


class TranscriptionBackend(ABC):
    """Abstract base class for transcription backends."""

    def __init__(self):
        self._lock = Lock()

    @abstractmethod
    def load_model(self, model_name: str, device: str, **kwargs) -> None:
        """Load the model. Called once at startup."""
        ...

    @abstractmethod
    def transcribe(self, audio_16k: np.ndarray, **kwargs) -> str:
        """
        Transcribe 16 kHz mono float32 audio to text.

        Thread-safe: implementations must use self._lock.
        """
        ...


class HuggingFaceBackend(TranscriptionBackend):
    """Transcription via HuggingFace transformers pipeline."""

    def __init__(self):
        super().__init__()
        self._pipe = None

    def load_model(self, model_name: str, device: str, **kwargs) -> None:
        from transformers import pipeline as hf_pipeline

        gen_kwargs = {}
        task = kwargs.get("task")
        language = kwargs.get("language")
        if task and task != "auto":
            gen_kwargs["task"] = task
        if language and language != "auto":
            gen_kwargs["language"] = language

        pipe_kwargs = dict(
            task="automatic-speech-recognition",
            model=model_name,
            device=device,
            generate_kwargs=gen_kwargs or None,
        )
        chunk_length = kwargs.get("pipe_chunk_length", 0)
        if chunk_length and chunk_length > 0:
            pipe_kwargs["chunk_length_s"] = chunk_length
            pipe_kwargs["ignore_warning"] = True

        with self._lock:
            self._pipe = hf_pipeline(**pipe_kwargs)
            # Clear forced_decoder_ids if set (can conflict with gen_kwargs)
            try:
                if gen_kwargs and hasattr(self._pipe, "model") and hasattr(self._pipe.model, "generation_config"):
                    if getattr(self._pipe.model.generation_config, "forced_decoder_ids", None) is not None:
                        self._pipe.model.generation_config.forced_decoder_ids = None
            except Exception:
                pass

        logging.info(f"HuggingFace backend loaded: {model_name} on {device}")

    def transcribe(self, audio_16k: np.ndarray, **kwargs) -> str:
        timestamps_mode = kwargs.get("timestamps_mode", "auto")
        no_ts_max_seconds = kwargs.get("no_ts_max_seconds", 29.9)

        dur = _seconds_of(audio_16k)
        want_ts = (timestamps_mode == "on") or (timestamps_mode == "auto" and dur >= 30.0)

        with self._lock:
            if self._pipe is None:
                logging.error("HuggingFace pipeline not loaded")
                return ""
            try:
                if not want_ts and dur > no_ts_max_seconds:
                    parts = _split_audio(audio_16k, no_ts_max_seconds)
                    texts = []
                    for p in parts:
                        out = self._pipe(p.copy(), batch_size=8)
                        texts.append(_extract_hf_text(out))
                    return " ".join(t.strip() for t in texts if t).lstrip()
                else:
                    if want_ts:
                        out = self._pipe(audio_16k.copy(), batch_size=8, return_timestamps=True)
                    else:
                        out = self._pipe(audio_16k.copy(), batch_size=8)
                    return _extract_hf_text(out)
            except Exception as e:
                logging.error(f"HuggingFace transcription error: {e}")
                play_effect('dump', blocking=False)
                return ""


class FasterWhisperBackend(TranscriptionBackend):
    """Transcription via faster-whisper (CTranslate2)."""

    def __init__(self):
        super().__init__()
        self._model = None

    def load_model(self, model_name: str, device: str, **kwargs) -> None:
        from faster_whisper import WhisperModel

        # Map torch device string to faster-whisper device/device_index
        fw_device = "cpu"
        device_index = 0
        if device.startswith("cuda"):
            fw_device = "cuda"
            parts = device.split(":")
            if len(parts) == 2:
                try:
                    device_index = int(parts[1])
                except ValueError:
                    pass

        # Map HuggingFace model name to faster-whisper size
        # e.g. "openai/whisper-medium" -> "medium"
        model_size = _hf_name_to_fw_size(model_name)

        compute_type = kwargs.get("compute_type", "float16" if fw_device == "cuda" else "int8")

        with self._lock:
            self._model = WhisperModel(
                model_size,
                device=fw_device,
                device_index=device_index,
                compute_type=compute_type,
            )

        logging.info(f"faster-whisper backend loaded: {model_size} on {fw_device}:{device_index} ({compute_type})")

    def transcribe(self, audio_16k: np.ndarray, **kwargs) -> str:
        task = kwargs.get("task", "transcribe")
        if task == "auto":
            task = "transcribe"
        language = kwargs.get("language")
        if language == "auto":
            language = None

        with self._lock:
            if self._model is None:
                logging.error("faster-whisper model not loaded")
                return ""
            try:
                segments, _info = self._model.transcribe(
                    audio_16k,
                    task=task,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
                text = " ".join(seg.text.strip() for seg in segments)
                return text.lstrip()
            except Exception as e:
                logging.error(f"faster-whisper transcription error: {e}")
                play_effect('dump', blocking=False)
                return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_hf_text(result_dict) -> str:
    """Normalize HuggingFace pipeline output to plain text."""
    try:
        if isinstance(result_dict, dict):
            if "text" in result_dict and result_dict["text"] is not None:
                return str(result_dict["text"]).lstrip()
            if "chunks" in result_dict and result_dict["chunks"]:
                joined = " ".join((ch.get("text", "") or "").strip() for ch in result_dict["chunks"])
                return joined.lstrip()
    except Exception:
        pass
    return ""


def _hf_name_to_fw_size(model_name: str) -> str:
    """
    Convert a HuggingFace model identifier like 'openai/whisper-medium'
    to a faster-whisper model size string like 'medium'.

    If the name doesn't match the openai/whisper-* pattern, return it as-is
    (faster-whisper also accepts model paths).
    """
    name = model_name.strip()
    # Handle "openai/whisper-large-v2" -> "large-v2"
    if "/" in name:
        suffix = name.split("/")[-1]
        if suffix.startswith("whisper-"):
            return suffix[len("whisper-"):]
        return suffix
    return name


def create_backend(backend_name: str) -> TranscriptionBackend:
    """Factory function to create a transcription backend by name."""
    name = backend_name.strip().lower()
    if name in ("faster-whisper", "faster_whisper", "fw"):
        try:
            import faster_whisper  # noqa: F401
            return FasterWhisperBackend()
        except ImportError:
            logging.warning("faster-whisper not installed; falling back to huggingface backend")
            return HuggingFaceBackend()
    elif name in ("huggingface", "hf", "transformers"):
        return HuggingFaceBackend()
    else:
        logging.warning(f"Unknown backend '{backend_name}'; using huggingface")
        return HuggingFaceBackend()
