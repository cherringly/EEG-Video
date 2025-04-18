# import csv
# import pandas as pd
# import numpy as np
# import scipy.signal as signal
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt, find_peaks
# import os
# from bionodebinopen import fn_BionodeBinOpen  # You must define this function separately


# expDay = '25-31-03'  # folder or name of the experiment
# fileN = 7  # File to inspect in the folder
# channel = 1  # Channel to look at
# fsBionode = (25e3)/2  # Sampling rate
# ADCres = 12  # ADC resolution
# blockPath = '/Users/maryzhang/Downloads/EarEEG_Matlab/Data/25-31-03/2025.03.31.17.53.13_AAO-18.bin'
# earData = fn_BionodeBinOpen(blockPath, ADCres, fsBionode)
# outputFolder = '/Users/maryzhang/Downloads/EarEEG_Matlab'  # output folder
# recording_duration_min = 10
# recording_duration_sec = recording_duration_min * 60  # Convert to seconds



# # Preprocessing
# highCutoff = 25  # frequency below which take the filter

# # Bionode
# rawChaB = np.array(earData['channelsData'])
# rawChaB = (rawChaB - 2**11) * 1.8 / (2**12 * 1000)  # Adjust using -60 dB and 12 bits for 1.8V
# bB, aB = butter(4, highCutoff / (fsBionode / 2), btype='low')
# PPfiltChaB = filtfilt(bB, aB, rawChaB[channel-1, :])
# # Constants
# fs = fsBionode  # Sampling rate
# win_sec = 0.5
# step_sec = 0.05
# win_samples = int(win_sec * fs)
# step_samples = int(step_sec * fs)

# window_size = int(60*fs)  # 60 seconds
# alpha_band = (8,12)  # Alpha band in Hz

# # Split signal into 1-minute chunks
# num_minutes = int(np.ceil(len(PPfiltChaB) / window_size))
# minutes = [PPfiltChaB[i * window_size : (i + 1) * window_size] 
#             for i in range(num_minutes)]

# # Frequency resolution setup
# freq_res = 1  # Hz
# nfft = int(fs)  # ensure 1 Hz resolution: fs / nfft = 1 Hz
# freqs = np.fft.rfftfreq(nfft, d=1/fs)
# freq_mask = (freqs >= 0) & (freqs <= 20)
# selected_freqs = freqs[freq_mask]

# # Sliding FFT
# log_power_matrix = []
# time_stamps = []

# for start in range(0, len(PPfiltChaB) - win_samples, step_samples):
#     end = start + win_samples
#     window = PPfiltChaB[start:end]

#     if len(window) < win_samples:
#         continue

#     windowed_signal = window * np.hamming(len(window))
#     fft_data = np.fft.rfft(windowed_signal, n=nfft)
#     power = np.abs(fft_data) ** 2
#     power_db = 10 * np.log10(power + 1e-12)  # avoid log(0)

#     log_power_matrix.append(power_db[freq_mask])
#     time_stamps.append(start / fs)

# # Save CSV
# csv_path = os.path.join(outputFolder, 'fft_sliding_0-20Hz.csv')
# with open(csv_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     header = ['Time (s)'] + [f'{f:.0f} Hz' for f in selected_freqs]
#     writer.writerow(header)

#     for t, row in zip(time_stamps, log_power_matrix):
#         writer.writerow([f'{t:.2f}'] + [f'{val:.2f}' for val in row])

# print(f"Sliding window FFT saved to: {csv_path}")

# timeB = np.arange(0, len(PPfiltChaB)) / fsBionode




# plt.figure()
# plt.suptitle("In Ear1")
# plt.subplot(2, 1, 1)
# plt.plot(timeB, rawChaB[channel-1, :])
# plt.xlabel("s")
# plt.ylabel("V")
# plt.title("Raw Data")

# plt.subplot(2, 1, 2)
# plt.plot(timeB, PPfiltChaB)
# plt.xlabel("s")
# plt.ylabel("V")
# plt.title("Preprocessed Data")
# plt.tight_layout()
# plt.show()



# # # Spectral Analysis
# # plt.figure(figsize=(12, 6))
# # plt.suptitle(f'Spectral Analysis (0-50 Hz) - Full {recording_duration_min} min Recording')
# # channelsBionode = [channel]

# # for i, ch in enumerate(channelsBionode):
# #     # Calculate spectrogram with parameters that ensure full duration coverage
# #     window_size_sec = 2  # Window size in seconds
# #     nperseg = int(fsBionode * window_size_sec)
# #     noverlap = int(nperseg * 0.9)  # 90% overlap
    
# #     # Calculate spectrogram
# #     f, t, Sxx = signal.spectrogram(PPfiltChaB, fs=fsBionode,
# #                                  nperseg=nperseg,
# #                                  noverlap=noverlap,
# #                                  scaling='density',
# #                                  mode='psd')
    
# #     # Verify time bins
# #     print(f"Spectrogram time bins: {len(t)}")
# #     print(f"Last time point: {t[-1]:.2f} seconds")
    
# #     # Filter frequencies
# #     freq_mask = (f >= 0.1) & (f <= highCutoff)
# #     f_filtered = f[freq_mask]
# #     Sxx_filtered = Sxx[freq_mask, :]
    
# #     # Plot
# #     plt.subplot(len(channelsBionode), 1, i+1)
# #     plt.pcolormesh(t / 60, f_filtered, 10 * np.log10(Sxx_filtered),
# #                  shading='gouraud', cmap='jet')
# #     plt.ylabel('Frequency [Hz]')
# #     plt.xlabel('Time [min]')
# #     plt.title(f'Spectrogram: Channel {ch}')
# #     plt.colorbar(label='Power/Frequency [dB/Hz]')
# #     plt.xlim(0, recording_duration_min)
    
# #     # Export to CSV
# #     spectrogram_db = 10 * np.log10(Sxx_filtered)
# #     df_spectrogram = pd.DataFrame(spectrogram_db,
# #                                index=np.round(f_filtered, 2),
# #                                columns=np.round(t, 2))
# #     output_filename = os.path.join(outputFolder, f'spectrogram_ch{ch}_10min.csv')
# #     df_spectrogram.to_csv(output_filename)
# #     print(f"Spectrogram data saved to {output_filename}")

# # plt.tight_layout()
# # plt.show()

# # print("Spectral Analysis Complete")




# # Compute alpha power for each minute
# print("Minute | Alpha Power (8-12 Hz) | Expected Pattern")
# print("-----------------------------------------------")
# for i, minute_signal in enumerate(minutes[:10]):  # First 10 minutes only
#     freqs, psd = signal.welch(minute_signal, fs, nperseg=1024)  # Smaller window for speed
#     alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
#     alpha_power = np.sum(psd[alpha_mask])  # Sum power in alpha band
    
#     # Check if minute is even/odd (1-based)
#     expected = "HIGH (even min)" if (i+1) % 2 == 0 else "LOW  (odd min)"
#     print(f"{i+1:>5} | {alpha_power:>18.2f} | {expected}")

import numpy as np
from scipy import signal
import time  # For debugging execution time

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

window_size = int(60*fs)  # 60 seconds
# ===== DEBUGGING ADDITIONS =====
import numpy as np
from scipy import signal
import time  # For debugging execution time

# [Keep your existing imports and data loading code]

# ===== DEBUGGING ADDITIONS =====
print("Starting alpha power analysis...")
start_time = time.time()

# Downsample the signal to 250 Hz (for faster processing)
downsample_factor = int(fsBionode // 250)
fs_downsampled = fsBionode / downsample_factor
signal_downsampled = PPfiltChaB[::downsample_factor]

print(f"Downsampled from {fsBionode} Hz to {fs_downsampled} Hz")
print(f"Signal length reduced from {len(PPfiltChaB)} to {len(signal_downsampled)} samples")

# Split into 1-minute chunks (using downsampled signal)
minute_samples = int(60 * fs_downsampled)
minutes = [signal_downsampled[i * minute_samples : (i + 1) * minute_samples] 
           for i in range(min(10, len(signal_downsampled) // minute_samples))]  # Only first 10 minutes

print(f"Processing {len(minutes)} minutes...")

# Compute alpha power for each minute
print("\nMinute | Alpha Power (8-12 Hz) | Expected Pattern")
print("-----------------------------------------------")

for i, minute_signal in enumerate(minutes):
    try:
        # Use Welch's method with minimal settings
        freqs, psd = signal.welch(minute_signal, fs_downsampled, 
                                 nperseg=min(256, len(minute_signal)),  # Prevent seg > data
                                 scaling='density')
        
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        alpha_power = np.trapz(psd[alpha_mask], freqs[alpha_mask])  # More accurate than sum
        
        expected = "HIGH (even min)" if (i+1) % 2 == 0 else "LOW (odd min)"
        print(f"{i+1:>6} | {alpha_power:>18.2f} | {expected}")
        
    except Exception as e:
        print(f"Error processing minute {i+1}: {str(e)}")

print(f"\nCompleted in {time.time() - start_time:.2f} seconds")