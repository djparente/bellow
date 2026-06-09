"""Delivery of transcribed text to the clipboard and/or as keystrokes."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class OutputSink:
    """Sends transcription results to the clipboard and emulated keyboard."""

    def __init__(self, use_clipboard: bool = True, use_keyboard: bool = True):
        if not use_clipboard and not use_keyboard:
            raise ValueError('At least one of clipboard or keyboard output must be enabled')
        self._use_clipboard = use_clipboard
        self._keyboard = None
        if use_keyboard:
            from pynput.keyboard import Controller
            self._keyboard = Controller()

    def deliver(self, text: str) -> None:
        """Deliver text to all enabled outputs. Failures in one output do not
        prevent the other from being attempted."""
        if not text:
            logger.info('Transcription was empty; nothing to deliver')
            return

        if self._use_clipboard:
            try:
                import pyperclip
                pyperclip.copy(text)
            except Exception as e:
                logger.error('Failed to copy to clipboard (on Linux, install xclip or xsel): %s', e)

        if self._keyboard is not None:
            try:
                self._keyboard.type(text)
            except Exception as e:
                logger.error('Failed to emulate keyboard input: %s', e)
