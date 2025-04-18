# # BionodeECGBPM.py
# # *Description*: This program loads Bionode files that are ECG and detects
# # the BPM





# BionodeECGBPM.py
# *Description*: This program loads Bionode files that are ECG and detects
# the BPM

import csv
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os
from bionodebinopen import fn_BionodeBinOpen  # You must define this function separately


expDay = '25-31-03'  # folder or name of the experiment
fileN = 7  # File to inspect in the folder
channel = 1  # Channel to look at
fsBionode = (25e3)/2  # Sampling rate
ADCres = 12  # ADC resolution
blockPath = '/Users/maryzhang/Downloads/EarEEG_Matlab/Data/25-31-03/2025.03.31.17.53.13_AAO-18.bin'
earData = fn_BionodeBinOpen(blockPath, ADCres, fsBionode)
outputFolder = '/Users/maryzhang/Downloads/EarEEG_Matlab'  # output folder
recording_duration_min = 10
recording_duration_sec = recording_duration_min * 60  # Convert to seconds



# Preprocessing
highCutoff = 25  # frequency below which take the filter

# Bionode
rawChaB = np.array(earData['channelsData'])
rawChaB = (rawChaB - 2**11) * 1.8 / (2**12 * 1000)  # Adjust using -60 dB and 12 bits for 1.8V
bB, aB = butter(4, highCutoff / (fsBionode / 2), btype='low')
PPfiltChaB = filtfilt(bB, aB, rawChaB[channel-1, :])
# Constants
fs = fsBionode  # Sampling rate
win_sec = 0.5
step_sec = 0.05
win_samples = int(win_sec * fs)
step_samples = int(step_sec * fs)

# Frequency resolution setup
freq_res = 1  # Hz
nfft = int(fs)  # ensure 1 Hz resolution: fs / nfft = 1 Hz
freqs = np.fft.rfftfreq(nfft, d=1/fs)
freq_mask = (freqs >= 0) & (freqs <= 50)
selected_freqs = freqs[freq_mask]

# Sliding FFT
log_power_matrix = []
time_stamps = []

for start in range(0, len(PPfiltChaB) - win_samples, step_samples):
    end = start + win_samples
    window = PPfiltChaB[start:end]

    if len(window) < win_samples:
        continue

    windowed_signal = window * np.hamming(len(window))
    fft_data = np.fft.rfft(windowed_signal, n=nfft)
    power = np.abs(fft_data) ** 2
    power_db = 10 * np.log10(power + 1e-12)  # avoid log(0)

    log_power_matrix.append(power_db[freq_mask])
    time_stamps.append(start / fs)

# Save CSV
csv_path = os.path.join(outputFolder, 'fft_sliding_0-50Hz.csv')
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['Time (s)'] + [f'{f:.0f} Hz' for f in selected_freqs]
    writer.writerow(header)

    for t, row in zip(time_stamps, log_power_matrix):
        writer.writerow([f'{t:.2f}'] + [f'{val:.2f}' for val in row])

print(f"Sliding window FFT saved to: {csv_path}")

timeB = np.arange(0, len(PPfiltChaB)) / fsBionode




plt.figure()
plt.suptitle("In Ear1")
plt.subplot(2, 1, 1)
plt.plot(timeB, rawChaB[channel-1, :])
plt.xlabel("s")
plt.ylabel("V")
plt.title("Raw Data")

plt.subplot(2, 1, 2)
plt.plot(timeB, PPfiltChaB)
plt.xlabel("s")
plt.ylabel("V")
plt.title("Preprocessed Data")
plt.tight_layout()
plt.show()



# Spectral Analysis
plt.figure(figsize=(12, 6))
plt.suptitle(f'Spectral Analysis (0-50 Hz) - Full {recording_duration_min} min Recording')
channelsBionode = [channel]

for i, ch in enumerate(channelsBionode):
    # Calculate spectrogram with parameters that ensure full duration coverage
    window_size_sec = 2  # Window size in seconds
    nperseg = int(fsBionode * window_size_sec)
    noverlap = int(nperseg * 0.9)  # 90% overlap
    
    # Calculate spectrogram
    f, t, Sxx = signal.spectrogram(PPfiltChaB, fs=fsBionode,
                                 nperseg=nperseg,
                                 noverlap=noverlap,
                                 scaling='density',
                                 mode='psd')
    
    # Verify time bins
    print(f"Spectrogram time bins: {len(t)}")
    print(f"Last time point: {t[-1]:.2f} seconds")
    
    # Filter frequencies
    freq_mask = (f >= 0.1) & (f <= highCutoff)
    f_filtered = f[freq_mask]
    Sxx_filtered = Sxx[freq_mask, :]
    
    # Plot
    plt.subplot(len(channelsBionode), 1, i+1)
    plt.pcolormesh(t / 60, f_filtered, 10 * np.log10(Sxx_filtered),
                 shading='gouraud', cmap='jet')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [min]')
    plt.title(f'Spectrogram: Channel {ch}')
    plt.colorbar(label='Power/Frequency [dB/Hz]')
    plt.xlim(0, recording_duration_min)
    
    # Export to CSV
    spectrogram_db = 10 * np.log10(Sxx_filtered)
    df_spectrogram = pd.DataFrame(spectrogram_db,
                               index=np.round(f_filtered, 2),
                               columns=np.round(t, 2))
    output_filename = os.path.join(outputFolder, f'spectrogram_ch{ch}_10min.csv')
    df_spectrogram.to_csv(output_filename)
    print(f"Spectrogram data saved to {output_filename}")

plt.tight_layout()
plt.show()

print("Spectral Analysis Complete")
