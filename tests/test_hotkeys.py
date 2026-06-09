"""Unit tests for hotkey specification parsing."""

import unittest

from bellow.hotkeys import to_pynput_spec


class HotkeySpecTests(unittest.TestCase):
    def test_default_toggle_hotkey(self):
        self.assertEqual(to_pynput_spec('ctrl+shift+alt+f11'), '<ctrl>+<shift>+<alt>+<f11>')

    def test_single_character_keys_stay_bare(self):
        self.assertEqual(to_pynput_spec('ctrl+shift+d'), '<ctrl>+<shift>+d')

    def test_aliases(self):
        self.assertEqual(to_pynput_spec('control+escape'), '<ctrl>+<esc>')
        self.assertEqual(to_pynput_spec('win+space'), '<cmd>+<space>')

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(to_pynput_spec(' Ctrl + Shift + F11 '), '<ctrl>+<shift>+<f11>')

    def test_pynput_format_passthrough(self):
        self.assertEqual(to_pynput_spec('<ctrl>+<alt>+h'), '<ctrl>+<alt>+h')

    def test_empty_spec_raises(self):
        with self.assertRaises(ValueError):
            to_pynput_spec('  ')


if __name__ == '__main__':
    unittest.main()
