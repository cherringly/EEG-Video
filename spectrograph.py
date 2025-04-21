import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os
import pandas as pd
from bionodebinopen import fn_BionodeBinOpen

expDay = '25-31-03'  # folder or name of the experiment
fileN = 7  # File to inspect in the folder
channel = 1  # Channel to look at
fsBionode = (25e3)/2  # Sampling rate
ADCres = 12  # ADC resolution
blockPath = '/Users/maryzhang/Downloads/EarEEG_Matlab/Data/25-31-03/2025.03.31.17.53.13_AAO-18.bin'
earData = fn_BionodeBinOpen(blockPath, ADCres, fsBionode)
outputFolder = '/Users/maryzhang/Downloads/EarEEG_Matlab'  # output folder
recording_duration_min = 10
recording_duration_sec = recording_duration_min * 60  # Convert to seconds\

#Preprocessing
highCutoff = 50  # frequency below which take the filter

# Bionode
rawChaB = np.array(earData['channelsData'])
rawChaB = (rawChaB - 2**11) * 1.8 / (2**12 * 1000)  # Adjust using -60 dB and 12 bits for 1.8V
bB, aB = butter(4, highCutoff / (fsBionode / 2), btype='low')
num_samples = int(recording_duration_sec * fsBionode)

# Verify data length
total_samples = rawChaB.shape[1]
expected_samples = int(recording_duration_sec * fsBionode)
print(f"Total samples in data: {total_samples}")
print(f"Expected samples for {recording_duration_min} min: {expected_samples}")


# Spectral Analysis using pspectrum (0–50 Hz High Resolution)
plt.figure()
plt.suptitle('Spectral Analysis (0–50 Hz)')
channelsBionode = [channel]  # Modify this if you have multiple channels

for i, ch in enumerate(channelsBionode):
    PPfiltChaB = filtfilt(bB, aB, earData['channelsData'][ch - 1, :])
    nperseg = int(fsBionode * 2)  # Larger window for better frequency resolution
    noverlap = int(nperseg * 0.9)  # 90% overlap as in MATLAB
    # f, t, Sxx = signal.spectrogram(PPfiltChaB, fs=fsBionode, nperseg=int(fsBionode/0.5), noverlap=int(fsBionode/0.5*0.9), scaling='spectrum')
    f, t, Sxx = signal.spectrogram(PPfiltChaB, fs=fsBionode, 
                                  nperseg=nperseg, 
                                  noverlap=noverlap,
                                  scaling='density')
    freq_mask = (f >= 0.1) & (f <= highCutoff)
    plt.subplot(len(channelsBionode), 1, i+1)
    plt.pcolormesh(t / 60, f[freq_mask], 10 * np.log10(Sxx[freq_mask, :]), shading='gouraud',cmap='jet')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (min)')
    plt.title(f'Spectrogram: Channel {ch}')
    plt.colorbar(label ='Power/Frequency [dB/Hz]')

plt.tight_layout()
# plt.set_cmap('jet')
plt.show()

print("Spectral Analysis Complete")

