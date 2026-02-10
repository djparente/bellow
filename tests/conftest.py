"""
Shared test fixtures and import workarounds.

pynput requires an X display on Linux, which isn't available in headless CI.
We mock it before any bellow.main import happens.
"""

import sys
from unittest.mock import MagicMock

# Mock sounddevice if PortAudio is not available (headless CI)
if "sounddevice" not in sys.modules:
    try:
        import sounddevice  # noqa: F401
    except OSError:
        sd_mock = MagicMock()
        sd_mock.InputStream = MagicMock
        sd_mock.default = MagicMock()
        sd_mock.query_devices = MagicMock(return_value=[])
        sd_mock.query_hostapis = MagicMock(return_value=[])
        sd_mock.check_input_settings = MagicMock()
        sys.modules["sounddevice"] = sd_mock

# Mock pynput.keyboard before anything imports bellow.main
if "pynput" not in sys.modules:
    pynput_mock = MagicMock()
    sys.modules["pynput"] = pynput_mock
    sys.modules["pynput.keyboard"] = pynput_mock.keyboard
    # Provide a Controller class that can be instantiated
    pynput_mock.keyboard.Controller = MagicMock
    pynput_mock.keyboard.GlobalHotKeys = MagicMock

# Mock clipboard too (may not be installed in CI)
if "clipboard" not in sys.modules:
    clipboard_mock = MagicMock()
    sys.modules["clipboard"] = clipboard_mock
