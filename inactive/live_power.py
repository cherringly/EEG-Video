# eeg_alpha_band_plot.py

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bionodebinopen import fn_BionodeBinOpen
import time

# Config
expDay = '25-31-03'
fileN = 7
channel = 1
fsBionode = 25e3 / 2  # 12.5 kHz
ADCres = 12
blockPath = '/Users/maryzhang/Downloads/EarEEG_Matlab/Data/25-31-03/2025.03.31.17.53.13_AAO-18.bin'
outputFolder = '/Users/maryzhang/Downloads/EarEEG_Matlab'
recording_duration_sec = 10 * 60

# Load EEG data
rawCha = np.array(fn_BionodeBinOpen(blockPath, ADCres, fsBionode)['channelsData'])
rawCha = (rawCha - 2**11) * 1.8 / (2**12 * 1000)  # Scale to volts
rawCha = np.nan_to_num(rawCha, nan=0.0, posinf=0.0, neginf=0.0)  # sanitize

# Low-pass filter for preprocessing
highCutoff = 50
sos_lp = signal.butter(4, highCutoff, btype='low', fs=fsBionode, output='sos')
eeg_signal = signal.sosfiltfilt(sos_lp, rawCha[channel - 1])

# Bandpass filter (Alpha: 8-12 Hz)
alpha_band = [8, 12]
sos_alpha = signal.butter(4, alpha_band, btype='bandpass', fs=fsBionode, output='sos')
eeg_alpha = signal.sosfiltfilt(sos_alpha, eeg_signal)

# Animation parameters
window_sec = 10
step_sec = 0.05  # faster updates
window_samples = int(window_sec * fsBionode)
step_samples = int(step_sec * fsBionode)

fig, ax = plt.subplots()
line, = ax.plot([], [], animated=True)

# Handle signal range safely
ymin, ymax = np.nanmin(eeg_alpha), np.nanmax(eeg_alpha)
if np.isnan(ymin) or np.isnan(ymax) or np.isinf(ymin) or np.isinf(ymax):
    raise ValueError("Invalid signal bounds for plotting")
ax.set_ylim(ymin, ymax)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.set_title('EEG Alpha Band (8-12 Hz)')

# Animation update logic
index = [0]
def update(frame):
    idx = index[0]
    if idx + window_samples > len(eeg_alpha):
        return line,
    segment = eeg_alpha[idx:idx + window_samples]
    start_time = idx / fsBionode
    time_window = np.arange(window_samples) / fsBionode + start_time
    line.set_data(time_window, segment)
    ax.set_xlim(time_window[0], time_window[-1])
    index[0] += step_samples
    return line,

ani = animation.FuncAnimation(fig, update, init_func=lambda: (line,), interval=step_sec * 1000, blit=True)
plt.tight_layout()
plt.show()