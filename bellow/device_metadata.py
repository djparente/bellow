import sounddevice as sd


def list_devices() -> None:
    """
    Prints a list of all input and output audio devices

    :return: None
    """
    devices = sd.query_devices()
    print(devices)


def get_device_name(idx: int | None = None, input_if_default: bool | None = None) -> str:
    """
    Returns the name of a device with index idx. If this is none, it returns the name of the
    default input or output device depending on the value of input_if_default. Both inputs cannot
    be simultaneously unspecified (None).

    :param idx: The index of the device in the query_devices list
    :param input_if_default: If idx is None, should we return the default input device (if false, return output device)
    :return: The name of the specified device
    """
    # If index is none, get one of the default devices
    if idx is None:
        if input_if_default is None:
            raise ValueError('Cannot retrieve input/output device if input_if_default is None')

        if input_if_default:
            return sd.query_devices(sd.default.device[0])['name']
        else:
            return sd.query_devices(sd.default.device[1])['name']

    # Otherwise return name by index
    devices = sd.query_devices()
    return devices[idx]['name']
