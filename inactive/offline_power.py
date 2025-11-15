# # import numpy as np
# # import matplotlib.pyplot as plt
# # from scipy import signal
# # from scipy.integrate import simpson as simps
# # from bionodebinopen import fn_BionodeBinOpen

# # def load_and_preprocess_data(block_path, adc_resolution, fs, channel):
# #     # Load binary EEG data and scale to volts
# #     data_dict = fn_BionodeBinOpen(block_path, adc_resolution, fs)
# #     raw_data = np.array(data_dict['channelsData'])
# #     raw_data.astype(np.float32)
# #     scale_factor = 1.8/4096.0
# #     # raw_data = (raw_data - 2**11) * 1.8 / (2**12)  # Convert ADC values to volts
# #     raw_data = (raw_data - 2048) * scale_factor
# #     raw_data = np.nan_to_num(raw_data)  # Replace NaNs with zero
# #     return raw_data[channel]  # Return data from specified channel

# # def print_data_stats(total_samples, fs):
# #     # Print basic stats about the data
# #     duration_sec = total_samples / fs
# #     print(f"Total samples: {total_samples}")
# #     print(f"Sampling rate: {fs} Hz")
# #     print(f"Total duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")

# # def bandpass_filter_alpha(data, fs):
# #     # Bandpass filter for alpha frequency band (8-12 Hz)
# #     # sos_alpha = signal.butter(4, [8, 12], btype='bandpass', fs=fs, output='sos')
# #     # filtered = signal.sosfiltfilt(sos_alpha, data)
# #     sos_bandpass = signal.butter(4, [0.1,25], btype='bandpass', fs=fs, output='sos')
# #     return signal.sosfiltfilt(sos_bandpass, data)
# # # low pass with 0.02 after (1/50 seconds window size)

# # def smooth_alpha_power(alpha_powers, fs, window_sec):
# #     """Apply low-pass filter to the alpha power time series."""
# #     # Compute effective sampling rate of the alpha power time series
# #     power_fs = 1 / window_sec  # e.g., 1 Hz for 1-second windows
    
# #     # Design low-pass filter (0.02 Hz cutoff)
# #     sos_low = signal.butter(4, 0.03, btype='lowpass', fs=power_fs, output='sos')
# #     smoothed_power = signal.sosfiltfilt(sos_low, alpha_powers)
    
# #     return smoothed_power
# # def compute_alpha_power(eeg_alpha, fs, window_sec):
# #     # Compute power in alpha band over successive 1-second windows
# #     window_samples = int(window_sec * fs)
# #     total_samples = len(eeg_alpha)
# #     n_windows = total_samples // window_samples

# #     print(f"Window size (samples): {window_samples}")
# #     print(f"Number of 1-second windows: {n_windows}")

# #     alpha_powers = []  # Alpha power per window
# #     time_minutes = []  # Time stamp in minutes for each window

# #     for i in range(n_windows):
# #         start = i * window_samples
# #         end = start + window_samples
# #         segment = eeg_alpha[start:end] * np.hanning(window_samples)  # Apply Hann window

# #         freqs, psd = signal.welch(
# #             segment,
# #             fs=fs,
# #             window='hann',
# #             nperseg=window_samples,
# #             noverlap=0,
# #             scaling='density'
# #         )

# #         alpha_mask = (freqs >= 8) & (freqs <= 12)  # Indices for alpha band
# #         alpha_power = simps(psd[alpha_mask], freqs[alpha_mask])
# #         alpha_powers.append(alpha_power)  # In V² due to input scaling
# #         time_minutes.append(i * window_sec / 60)

# #     return np.array(time_minutes), np.array(alpha_powers)

# # def plot_alpha_power(time_minutes, alpha_powers, smoothed_power=None):
# #     # Plot alpha power over time and show per-minute average
# #     plt.figure(figsize=(12, 6))
# #     plt.plot(time_minutes, alpha_powers, color='green', alpha=0.5, label='Alpha Power (raw)')
    
# #     if smoothed_power is not None:
# #         plt.plot(time_minutes, smoothed_power, color='red', linewidth=2, label='Alpha Power (smoothed)')
        
# #     total_minutes = int(time_minutes[-1]) + 1
# #     for m in range(total_minutes):
# #         idx = (time_minutes >= m) & (time_minutes < m + 1)
# #         if np.any(idx):
# #             avg_power = np.mean(alpha_powers[idx])  # Average alpha power for the minute
# #             print(f"Minute {m}: Avg Alpha Power = {avg_power:.2e} V²")
# #             plt.hlines(avg_power, m, m + 1, colors='blue', linestyles='dashed', linewidth=2, label='1-min Avg' if m == 0 else "")

# #     plt.xlabel('Time (minutes)')
# #     plt.ylabel('Alpha Power (V²)')
# #     plt.yscale('log')  
# #     plt.title('Alpha Power Over Time (8–12 Hz)')
# #     plt.grid(True)
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.show()

# # def main():
# #     channel = 1  # Channel index to analyze
# #     ADCres = 12  # ADC resolution (bits)
# #     fsBionode = 5537  # Sampling rate in Hz
# #     window_sec = 1  # Window size in seconds
# #     blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"  # File path

# #     raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
# #     print_data_stats(len(raw_channel_data), fsBionode)

# #     eeg_alpha = bandpass_filter_alpha(raw_channel_data, fsBionode)
# #     time_minutes, alpha_powers = compute_alpha_power(eeg_alpha, fsBionode, window_sec)
    
# #     # Smooth the alpha power time series (not the raw EEG)
# #     smoothed_power = smooth_alpha_power(alpha_powers, fsBionode, window_sec)
    
# #     plot_alpha_power(time_minutes, alpha_powers, smoothed_power)

# #     print(f"First 5 time points (min): {time_minutes[:5]}")
# #     print(f"Last time point (min): {time_minutes[-1]:.2f}")
# #     print(f"Number of points: {len(time_minutes)}")


# # if __name__ == "__main__":
# #     main()














# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.integrate import simpson as simps
# import pandas as pd
# from bionodebinopen import fn_BionodeBinOpen

# def load_and_preprocess_data(block_path, adc_resolution, fs, channel):
#     data_dict = fn_BionodeBinOpen(block_path, adc_resolution, fs)
#     raw_data = np.array(data_dict['channelsData'])
#     scale_factor = 1.8 / 4096.0
#     raw_data = (raw_data - 2048) * scale_factor
#     raw_data = np.nan_to_num(raw_data)
#     return raw_data[channel]

# def print_data_stats(total_samples, fs):
#     duration_sec = total_samples / fs
#     print(f"Total samples: {total_samples}")
#     print(f"Sampling rate: {fs} Hz")
#     print(f"Total duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")

# def bandpass_filter_alpha(data, fs):
#     sos_bandpass = signal.butter(4, [0.1, 25], btype='bandpass', fs=fs, output='sos')
#     return signal.sosfiltfilt(sos_bandpass, data)

# def smooth_alpha_power(alpha_powers, fs, window_sec):
#     power_fs = 1 / window_sec
#     sos_low = signal.butter(4, 0.03, btype='lowpass', fs=power_fs, output='sos')
#     return signal.sosfiltfilt(sos_low, alpha_powers)

# def compute_alpha_power(eeg_alpha, fs, window_sec):
#     window_samples = int(window_sec * fs)
#     total_samples = len(eeg_alpha)
#     n_windows = total_samples // window_samples

#     alpha_powers = []
#     time_minutes = []

#     for i in range(n_windows):
#         start = i * window_samples
#         end = start + window_samples
#         segment = eeg_alpha[start:end] * np.hanning(window_samples)

#         freqs, psd = signal.welch(segment, fs=fs, window='hann', nperseg=window_samples, noverlap=0, scaling='density')
#         alpha_mask = (freqs >= 8) & (freqs <= 12)
#         alpha_power = simps(psd[alpha_mask], freqs[alpha_mask])
#         alpha_powers.append(alpha_power)
#         time_minutes.append(i * window_sec / 60)

#     return np.array(time_minutes), np.array(alpha_powers)

# def compute_eye_state_vector_events(csv_path):
#     MIN_EYE_CLOSED_FRAMES = 90
#     df = pd.read_csv(csv_path)
#     eye_states = df['Eye State'].values
#     timestamps = df['Timestamp (s)'].values

#     event_indices = []
#     i = 0
#     n = len(eye_states)

#     while i < n:
#         if eye_states[i] == 'CLOSED':
#             start = i
#             while i < n and eye_states[i] == 'CLOSED':
#                 i += 1
#             closed_duration = i - start
#             if closed_duration >= MIN_EYE_CLOSED_FRAMES:
#                 j = start
#                 while (j + 1) < n:
#                     if eye_states[j] == 'OPEN' and eye_states[j+1] == 'OPEN':
#                         break
#                     else:
#                         closed_duration += 1
#                     j += 1
#                 event_indices.append((start, j))
#                 i = j
#         else:
#             i += 1

#     return event_indices, timestamps

# def compute_avg_alpha_for_events(alpha_powers, window_sec, event_indices, timestamps):
#     power_fs = 1 / window_sec
#     closed_powers = []
#     open_powers = []

#     for i, (start, end) in enumerate(event_indices):
#         t_start = timestamps[start]
#         t_end = timestamps[end]
#         idx_start = int(t_start * power_fs)
#         idx_end = int(t_end * power_fs)
#         closed_powers.append(np.mean(alpha_powers[idx_start:idx_end]))

#         if i > 0:
#             prev_end = event_indices[i-1][1]
#             t_prev_end = timestamps[prev_end]
#             idx_prev_end = int(t_prev_end * power_fs)
#             idx_start_open = idx_prev_end
#             idx_end_open = int(t_start * power_fs)
#             open_powers.append(np.mean(alpha_powers[idx_start_open:idx_end_open]))

#     return closed_powers, open_powers

# def plot_alpha_power(time_minutes, alpha_powers, smoothed_power=None):
#     plt.figure(figsize=(12, 6))
#     plt.plot(time_minutes, alpha_powers, color='green', alpha=0.5, label='Alpha Power (raw)')
#     if smoothed_power is not None:
#         plt.plot(time_minutes, smoothed_power, color='red', linewidth=2, label='Alpha Power (smoothed)')
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Alpha Power (V²)')
#     plt.yscale('log')
#     plt.title('Alpha Power Over Time (8–12 Hz)')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def main():
#     channel = 1
#     ADCres = 12
#     fsBionode = 5537
#     window_sec = 1
#     blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
#     eye_csv = r"\Users\maryz\EEG-Video\annotations\eye_states.csv"

#     raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
#     print_data_stats(len(raw_channel_data), fsBionode)

#     eeg_alpha = bandpass_filter_alpha(raw_channel_data, fsBionode)
#     time_minutes, alpha_powers = compute_alpha_power(eeg_alpha, fsBionode, window_sec)

#     smoothed_power = smooth_alpha_power(alpha_powers, fsBionode, window_sec)
#     plot_alpha_power(time_minutes, alpha_powers, smoothed_power)

#     event_indices, timestamps = compute_eye_state_vector_events(eye_csv)
#     closed_powers, open_powers = compute_avg_alpha_for_events(alpha_powers, window_sec, event_indices, timestamps)

#     print("\nClosed Eye Periods Avg Power:")
#     for i, power in enumerate(closed_powers):
#         print(f"  Event {i+1}: {power:.2e} V²")

#     print("\nOpen Eye Periods Avg Power:")
#     for i, power in enumerate(open_powers):
#         print(f"  Open Epoch {i+1}: {power:.2e} V²")

# if __name__ == "__main__":
#     main()






# PURE ALPHA POWER GRAPH ---------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.integrate import simpson as simps
# from bionodebinopen import fn_BionodeBinOpen

# def load_and_preprocess_data(block_path, adc_resolution, fs, channel):
#     # Load binary EEG data and scale to volts
#     data_dict = fn_BionodeBinOpen(block_path, adc_resolution, fs)
#     raw_data = np.array(data_dict['channelsData'])
#     raw_data.astype(np.float32)
#     scale_factor = 1.8/4096.0
#     raw_data = (raw_data - 2048) * scale_factor
#     raw_data = np.nan_to_num(raw_data)  # Replace NaNs with zero
#     return raw_data[channel]  # Return data from specified channel

# def print_data_stats(total_samples, fs):
#     # Print basic stats about the data
#     duration_sec = total_samples / fs
#     print(f"Total samples: {total_samples}")
#     print(f"Sampling rate: {fs} Hz")
#     print(f"Total duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")

# def bandpass_filter_alpha(data, fs):
#     # Bandpass filter for alpha frequency band (8-12 Hz)
#     sos_bandpass = signal.butter(4, [0.1, 25], btype='bandpass', fs=fs, output='sos')
#     return signal.sosfiltfilt(sos_bandpass, data)

# def compute_alpha_power(eeg_alpha, fs, window_sec):
#     # Compute power in alpha band over successive 1-second windows
#     window_samples = int(window_sec * fs)
#     total_samples = len(eeg_alpha)
#     n_windows = total_samples // window_samples

#     print(f"Window size (samples): {window_samples}")
#     print(f"Number of 1-second windows: {n_windows}")

#     alpha_powers = []  # Alpha power per window
#     time_minutes = []  # Time stamp in minutes for each window

#     for i in range(n_windows):
#         start = i * window_samples
#         end = start + window_samples
#         segment = eeg_alpha[start:end] * np.hanning(window_samples)  # Apply Hann window

#         freqs, psd = signal.welch(
#             segment,
#             fs=fs,
#             window='hann',
#             nperseg=window_samples,
#             noverlap=0,
#             scaling='density'
#         )

#         alpha_mask = (freqs >= 8) & (freqs <= 12)  # Indices for alpha band
#         alpha_power = simps(psd[alpha_mask], freqs[alpha_mask])
#         alpha_powers.append(alpha_power)  # In V² due to input scaling
#         time_minutes.append(i * window_sec / 60)

#     return np.array(time_minutes), np.array(alpha_powers)

# def plot_alpha_power(time_minutes, alpha_powers):
#     # Plot alpha power over time
#     plt.figure(figsize=(12, 6))
#     plt.plot(time_minutes, alpha_powers, color='green', alpha=0.7, label='Alpha Power')
    
#     plt.xlabel('Time (minutes)')
#     plt.ylabel('Alpha Power (V²)')
#     plt.yscale('log')  
#     plt.title('Alpha Power Over Time (8–12 Hz)')
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def main():
#     channel = 1  # Channel index to analyze
#     ADCres = 12  # ADC resolution (bits)
#     fsBionode = 5537  # Sampling rate in Hz
#     window_sec = 1  # Window size in seconds
#     blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"  # File path

#     raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
#     print_data_stats(len(raw_channel_data), fsBionode)

#     eeg_alpha = bandpass_filter_alpha(raw_channel_data, fsBionode)
#     time_minutes, alpha_powers = compute_alpha_power(eeg_alpha, fsBionode, window_sec)
    
#     plot_alpha_power(time_minutes, alpha_powers)

#     print(f"First 5 time points (min): {time_minutes[:5]}")
#     print(f"Last time point (min): {time_minutes[-1]:.2f}")
#     print(f"Number of points: {len(time_minutes)}")

# if __name__ == "__main__":
#     main()



import numpy as np
import matplotlib.pyplot as plt
from bionodebinopen import fn_BionodeBinOpen

def load_and_preprocess_data(block_path, adc_resolution, fs, channel):
    # Load binary EEG data and scale to volts
    data_dict = fn_BionodeBinOpen(block_path, adc_resolution, fs)
    raw_data = np.array(data_dict['channelsData'])
    raw_data.astype(np.float32)
    scale_factor = 1.8/4096.0
    raw_data = (raw_data - 2048) * scale_factor
    return np.nan_to_num(raw_data[channel])  # Return data from specified channel

def print_data_stats(total_samples, fs):
    duration_sec = total_samples / fs
    print(f"Total samples: {total_samples}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Total duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")

def downsample_for_plotting(data, original_fs, target_fs=200):
    """Downsample data to make plotting faster while preserving shape"""
    step = int(original_fs/target_fs)
    return data[::step]

def plot_raw_eeg_voltage(raw_data, fs, duration_minutes=3):
    duration_seconds = duration_minutes * 60
    num_samples = int(duration_seconds * fs)
    
    # Downsample the data for plotting
    plot_fs = 200  # Target sampling rate for plotting
    downsampled_data = downsample_for_plotting(raw_data[:num_samples], fs, plot_fs)
    downsampled_time = np.arange(0, len(downsampled_data))/plot_fs
    
    plt.figure(figsize=(12, 6))
    plt.plot(downsampled_time, downsampled_data, 
            color='blue', alpha=0.7, linewidth=0.5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Raw EEG Voltage (Downsampled to {plot_fs} Hz) - First {duration_minutes} Minutes')
    plt.tight_layout()
    
    # Enable interactive mode for better responsiveness
    plt.ion()
    plt.show()
    plt.pause(0.001)  # Needed for some backends to display properly

def main():
    channel = 1  # Channel index to analyze
    ADCres = 12  # ADC resolution (bits)
    fsBionode = 5537  # Sampling rate in Hz
    blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"  # File path

    print("Loading data...")
    raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
    print_data_stats(len(raw_channel_data), fsBionode)

    print("Plotting data...")
    plot_raw_eeg_voltage(raw_channel_data, fsBionode, duration_minutes=3)
    
    # Keep the plot window open
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()