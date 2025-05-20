import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import simpson as simps
from bionodebinopen import fn_BionodeBinOpen

def load_and_preprocess_data(block_path, adc_resolution, fs, channel):
    # Load binary EEG data and scale to volts
    data_dict = fn_BionodeBinOpen(block_path, adc_resolution, fs)
    raw_data = np.array(data_dict['channelsData'])
    raw_data = (raw_data - 2**11) * 1.8 / (2**12)  # Convert ADC values to volts
    raw_data = np.nan_to_num(raw_data)  # Replace NaNs with zero
    return raw_data[channel]  # Return data from specified channel

def print_data_stats(total_samples, fs):
    # Print basic stats about the data
    duration_sec = total_samples / fs
    print(f"Total samples: {total_samples}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Total duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")

def bandpass_filter_alpha(data, fs):
    # Bandpass filter for alpha frequency band (8-12 Hz)
    sos_alpha = signal.butter(4, [8, 12], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos_alpha, data)

def compute_alpha_power(eeg_alpha, fs, window_sec):
    # Compute power in alpha band over successive 1-second windows
    window_samples = int(window_sec * fs)
    total_samples = len(eeg_alpha)
    n_windows = total_samples // window_samples

    print(f"Window size (samples): {window_samples}")
    print(f"Number of 1-second windows: {n_windows}")

    alpha_powers = []  # Alpha power per window
    time_minutes = []  # Time stamp in minutes for each window

    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        segment = eeg_alpha[start:end] * np.hanning(window_samples)  # Apply Hann window

        freqs, psd = signal.welch(
            segment,
            fs=fs,
            window='hann',
            nperseg=window_samples,
            noverlap=0,
            scaling='density'
        )

        alpha_mask = (freqs >= 8) & (freqs <= 12)  # Indices for alpha band
        alpha_power = simps(psd[alpha_mask], freqs[alpha_mask])
        alpha_powers.append(alpha_power)  # In V² due to input scaling
        time_minutes.append(i * window_sec / 60)

    return np.array(time_minutes), np.array(alpha_powers)

def plot_alpha_power(time_minutes, alpha_powers):
    # Plot alpha power over time and show per-minute average
    plt.figure(figsize=(12, 6))
    plt.plot(time_minutes, alpha_powers, color='green', label='Alpha Power')

    total_minutes = int(time_minutes[-1]) + 1
    for m in range(total_minutes):
        idx = (time_minutes >= m) & (time_minutes < m + 1)
        if np.any(idx):
            avg_power = np.mean(alpha_powers[idx])  # Average alpha power for the minute
            print(f"Minute {m}: Avg Alpha Power = {avg_power:.2e} V²")
            plt.hlines(avg_power, m, m + 1, colors='blue', linestyles='dashed', linewidth=2, label='1-min Avg' if m == 0 else "")

    plt.xlabel('Time (minutes)')
    plt.ylabel('Alpha Power (V²)')
    plt.title('Alpha Power Over Time (8–12 Hz)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    channel = 1  # Channel index to analyze
    ADCres = 12  # ADC resolution (bits)
    fsBionode = 5537  # Sampling rate in Hz
    window_sec = 1  # Window size in seconds
    blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"  # File path

    raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
    print_data_stats(len(raw_channel_data), fsBionode)

    eeg_alpha = bandpass_filter_alpha(raw_channel_data, fsBionode)
    time_minutes, alpha_powers = compute_alpha_power(eeg_alpha, fsBionode, window_sec)

    print(f"First 5 time points (min): {time_minutes[:5]}")
    print(f"Last time point (min): {time_minutes[-1]:.2f}")
    print(f"Number of points: {len(time_minutes)}")

    plot_alpha_power(time_minutes, alpha_powers)

if __name__ == "__main__":
    main()