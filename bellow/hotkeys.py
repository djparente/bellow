"""Global hotkey handling for Bellow, built on pynput.

pynput was chosen over the `keyboard` package because `keyboard` requires
root privileges on Linux (it reads /dev/input directly). pynput listens via
X11 (XRecord) on Linux and Win32 hooks on Windows, neither of which needs
elevated privileges.

Hotkeys are written in the friendly "ctrl+shift+alt+f11" style used by
Bellow's CLI and converted to pynput's "<ctrl>+<shift>+<alt>+<f11>" syntax.

Limitations: on Wayland sessions pynput cannot observe global key events;
run under X11 (or an XWayland-hosted session) for global hotkeys to work.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Callable

logger = logging.getLogger(__name__)

# Friendly aliases mapped onto pynput key names
_ALIASES = {
    'control': 'ctrl',
    'escape': 'esc',
    'return': 'enter',
    'win': 'cmd',
    'windows': 'cmd',
    'super': 'cmd',
    'command': 'cmd',
    'menu': 'menu',
    'capslock': 'caps_lock',
    'pageup': 'page_up',
    'pagedown': 'page_down',
}


def to_pynput_spec(spec: str) -> str:
    """Convert 'ctrl+shift+alt+f11' to pynput's '<ctrl>+<shift>+<alt>+<f11>'.

    Specs already containing '<' are assumed to be in pynput format and are
    passed through unchanged.
    """
    spec = spec.strip()
    if '<' in spec:
        return spec.lower()
    parts = [p.strip().lower() for p in spec.split('+') if p.strip()]
    if not parts:
        raise ValueError(f'Empty hotkey specification: {spec!r}')
    converted = []
    for part in parts:
        part = _ALIASES.get(part, part)
        # Single printable characters stay bare; named keys get <> brackets
        converted.append(part if len(part) == 1 else f'<{part}>')
    return '+'.join(converted)


def warn_if_wayland() -> None:
    """Emit a clear warning when running under a Wayland session."""
    if sys.platform.startswith('linux') and os.environ.get('XDG_SESSION_TYPE', '').lower() == 'wayland':
        logger.warning(
            'You appear to be running a Wayland session. Global hotkeys require X11; '
            'they will likely not work here. Log into an X11 session to use Bellow.')


class HotkeyManager:
    """Registers a set of global hotkeys and dispatches their callbacks.

    pynput's GlobalHotKeys runs all callbacks sequentially on its listener
    thread, so callbacks never race with one another; keep them quick.
    """

    def __init__(self, bindings: dict[str, Callable[[], None]]):
        """
        :param bindings: mapping of friendly hotkey spec -> zero-argument callback
        """
        from pynput import keyboard as pk

        converted: dict[str, Callable[[], None]] = {}
        for spec, callback in bindings.items():
            pyn_spec = to_pynput_spec(spec)
            pk.HotKey.parse(pyn_spec)  # validate early, fail at startup not at keypress
            if pyn_spec in converted:
                raise ValueError(f'Duplicate hotkey: {spec!r} ({pyn_spec})')
            converted[pyn_spec] = self._wrap(spec, callback)

        self._listener = pk.GlobalHotKeys(converted)

    @staticmethod
    def _wrap(spec: str, callback: Callable[[], None]) -> Callable[[], None]:
        def safe_callback():
            try:
                callback()
            except Exception:
                logger.exception('Unhandled error in hotkey handler for %s', spec)
        return safe_callback

    def start(self) -> None:
        self._listener.start()
        # wait() raises listener thread startup errors (e.g. no X display) here
        self._listener.wait()

    def stop(self) -> None:
        self._listener.stop()
