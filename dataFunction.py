
import numpy as np
from scipy.signal import butter, filtfilt
from bionodebinopen import fn_BionodeBinOpen
import time

def load_and_filter(filename, ADCres=12, sampR=5537, filter_config=None):
    """
    Load and optionally filter BioNode .bin data.

    Args:
        filename (str): Path to .bin file.
        ADCres (int): ADC resolution (typically 12 or 24).
        sampR (int): Sampling rate in Hz.
        filter_config (dict): Optional filter configuration per channel.
            Format: {
                0: {"type": "low", "cutoff": 50},
                1: {"type": "bandpass", "cutoff": [10, 1000]},
                ...
            }

    Returns:
        dict: {
            "raw": raw_scaled (channels x time),
            "filtered": filtered (channels x time),
            "time": time array,
            "fs": sample rate,
            "numChannels": number of channels
        }
    """
    unpacked = fn_BionodeBinOpen(filename, ADCres, sampR)
    raw = unpacked["channelsData"].astype(np.float32)
    numChannels = unpacked["numChannels"]
    fs = unpacked["sampleRate"]
    time_arr = unpacked["time"]

    if ADCres == 12:
        raw_scaled = (raw - 2**11) * 1.8 / (2**12 * 1000)
    elif ADCres == 24:
        raw_scaled = (raw - 2**23) * 1.8 / (2**24 * 1000)
    else:
        raise ValueError(f"Unsupported ADC resolution: {ADCres}")

    filtered = np.zeros_like(raw_scaled)
    for ch in range(numChannels):
        if filter_config and ch in filter_config:
            config = filter_config[ch]
            if config["type"] == "low":
                b, a = butter(4, config["cutoff"] / (fs / 2), btype='low')
            elif config["type"] == "high":
                b, a = butter(4, config["cutoff"] / (fs / 2), btype='high')
            elif config["type"] == "bandpass":
                low, high = config["cutoff"]
                b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='bandpass')
            else:
                raise ValueError(f"Unsupported filter type: {config['type']}")
            filtered[ch] = filtfilt(b, a, raw_scaled[ch])
        else:
            filtered[ch] = raw_scaled[ch]

    return {
        "raw": raw_scaled,
        "filtered": filtered,
        "time": time_arr,
        "fs": fs,
        "numChannels": numChannels
    }

def sync_to_wall_clock(vid_time, t0_real):
    """
    Synchronize processing with real-world time.

    Args:
        vid_time (float): Time (in seconds) since the video started
        t0_real (list): Mutable list containing [None] or [start_time]

    Returns:
        float: Sleep duration (0 if no sleep needed)
    """
    if t0_real[0] is None:
        t0_real[0] = time.time() - vid_time

    to_sleep = t0_real[0] + vid_time - time.time()
    if to_sleep > 0:
        time.sleep(to_sleep)
    return to_sleep
