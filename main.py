import numpy as np
from scipy import signal
import time  # For debugging execution time
import pandas as pd
import matplotlib.pyplot as plt
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
highCutoff = 20  # frequency below which take the filter
# TODO: Isn't this a lowpass filter, not highpass?

# Load and preprocess data
rawChaB = np.array(earData['channelsData'])
rawChaB = (rawChaB - 2**11) * 1.8 / (2**12 * 1000)
bB, aB = signal.butter(4, highCutoff / (fsBionode / 2), btype='low')
PPfiltChaB = signal.filtfilt(bB, aB, rawChaB[channel-1, :])

# STFT Parameters
fs = fsBionode
win_sec = 0.5  # Window size in seconds
step_sec = 0.05  # Step size in seconds
win_samples = int(win_sec * fs)
step_samples = int(step_sec * fs)
nperseg = win_samples
noverlap = win_samples - step_samples

# Compute STFT
print("Computing STFT...")
start_time = time.time()
f, t, Zxx = signal.stft(PPfiltChaB, fs=fs, nperseg=nperseg, noverlap=noverlap)
power = np.abs(Zxx)**2  # Convert to power spectral density

# Filter frequencies of interest (0-20 Hz)
freq_mask = (f >= 0.1) & (f <= highCutoff)
f_filtered = f[freq_mask]
power_filtered = power[freq_mask, :]

print(f"STFT computed in {time.time() - start_time:.2f} seconds")

# Create output directory if it doesn't exist
os.makedirs(outputFolder, exist_ok=True)

# Export STFT results to CSV
output_path = os.path.join(outputFolder, 'stft_results.csv')
print(f"Exporting STFT results to {output_path}")

# Prepare DataFrame for CSV export
stft_df = pd.DataFrame(power_filtered.T, 
                      index=t, 
                      columns=[f"{freq:.1f} Hz" for freq in f_filtered])
stft_df.index.name = 'Time (s)'
stft_df.to_csv(output_path)

# Plot the spectrogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f_filtered, 10 * np.log10(power_filtered),
             shading='gouraud', cmap='jet')
plt.colorbar(label='Power/Frequency [dB/Hz]')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title(f'Short-Time Fourier Transform (STFT)\nWindow: {win_sec}s, Step: {step_sec}s')
plt.tight_layout()

# Save the plot
plot_path = os.path.join(outputFolder, 'stft_spectrogram.png')
plt.savefig(plot_path)
print(f"Spectrogram saved to {plot_path}")
plt.show()

















# Alpha Power Analysis
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

timeB = np.arange(0, len(PPfiltChaB)) / fsBionode

# TODO: for some reason, when plotting preprocessed data, spectral analysis doesn't show up
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


plt.tight_layout()
plt.show()

print("Spectral Analysis Complete")


print(f"\nCompleted in {time.time() - start_time:.2f} seconds")