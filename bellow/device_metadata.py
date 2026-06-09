"""Helpers for inspecting audio devices."""

from __future__ import annotations

import sounddevice as sd


def list_devices() -> None:
    """Prints a list of all input and output audio devices."""
    print(sd.query_devices())


def get_device_name(device: int | str | None = None, kind: str = 'input') -> str:
    """Return the name of an audio device.

    :param device: device index or name substring; None for the default device
    :param kind: 'input' or 'output'; used to pick the default device when
        device is None
    :return: the device name, or a description of the failure
    """
    try:
        return sd.query_devices(device, kind if device is None else None)['name']
    except Exception as e:
        return f'<unavailable: {e}>'
