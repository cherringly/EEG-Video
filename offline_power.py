import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import simpson as simps
from bionodebinopen import fn_BionodeBinOpen

channel = 0
ADCres = 12
fsBionode = 5537 
window_sec = 1
blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"

data_dict = fn_BionodeBinOpen(blockPath, ADCres, fsBionode)
rawCha = np.array(data_dict['channelsData'])
rawCha = (rawCha - 2**11) * 1.8 / (2**12)
rawCha = np.nan_to_num(rawCha)

total_samples = rawCha.shape[1]
total_duration_sec = total_samples / fsBionode
print(f"Total samples: {total_samples}")
print(f"Sampling rate: {fsBionode} Hz")
print(f"Total duration: {total_duration_sec:.2f} seconds ({total_duration_sec/60:.2f} minutes)")

# band pass filter for alpha band (8-12 Hz)
sos_alpha = signal.butter(4, [8, 12], btype='bandpass', fs=fsBionode, output='sos')
eeg_alpha = signal.sosfiltfilt(sos_alpha, rawCha[channel])


window_samples = int(window_sec * fsBionode) # number of samples in 1 second window
n_windows = total_samples // window_samples # number of 1-second windows
print(f"Window size (samples): {window_samples}")
print(f"Number of 1-second windows: {n_windows}")

alpha_powers = []
time_minutes = []

for i in range(n_windows):
    start = i * window_samples
    end = start + window_samples
    segment = eeg_alpha[start:end] * np.hanning(window_samples)

    freqs, psd = signal.welch(
        segment,
        fs=fsBionode,
        window='hann',
        nperseg=window_samples,
        noverlap=0,
        scaling='density'
    )

    alpha_mask = (freqs >= 8) & (freqs <= 12)
    alpha_power = simps(psd[alpha_mask], freqs[alpha_mask])
    alpha_powers.append(alpha_power * 1e6)  # μV²
    time_minutes.append(i * window_sec / 60)

print(f"First 5 time points (min): {time_minutes[:5]}")
print(f"Last time point (min): {time_minutes[-1]:.2f}")
print(f"Number of points: {len(time_minutes)}")

plt.figure(figsize=(12, 6))
plt.plot(time_minutes, alpha_powers, color='green', label='Alpha Power')

alpha_powers = np.array(alpha_powers) # alpha power in μV²
time_minutes = np.array(time_minutes) # time in minutes
total_minutes = int(time_minutes[-1]) + 1

# Loop through each minute and calculate the average power
for m in range(total_minutes):
    idx = (time_minutes >= m) & (time_minutes < m + 1) #
    if np.any(idx):
        avg_power = np.mean(alpha_powers[idx]) # average power in μV²
        plt.hlines(avg_power, m, m + 1, colors='blue', linestyles='dashed', linewidth=2, label='1-min Avg' if m == 0 else "")

plt.xlabel('Time (minutes)')
plt.ylabel('Alpha Power (μV²)')
plt.title('Alpha Power Over Time (8–12 Hz)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
