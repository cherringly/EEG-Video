import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import simpson as simps
import pandas as pd
from bionodebinopen import fn_BionodeBinOpen

def load_and_preprocess_data(block_path, adc_resolution, fs, channel):
    data_dict = fn_BionodeBinOpen(block_path, adc_resolution, fs)
    raw_data = np.array(data_dict['channelsData'])
    scale_factor = 1.8 / 4096.0
    raw_data = (raw_data - 2048) * scale_factor
    raw_data = np.nan_to_num(raw_data)
    return raw_data[channel]

def print_data_stats(total_samples, fs):
    duration_sec = total_samples / fs
    print(f"Total samples: {total_samples}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Total duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")
    return duration_sec

def bandpass_filter_alpha(data, fs):
    sos_bandpass = signal.butter(4, [0.1, 25], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos_bandpass, data)

def smooth_alpha_power(alpha_powers, fs, window_sec):
    power_fs = 1 / window_sec
    sos_low = signal.butter(4, 0.025, btype='lowpass', fs=power_fs, output='sos')
    return signal.sosfiltfilt(sos_low, alpha_powers)

def compute_alpha_power(eeg_alpha, fs, window_sec):
    window_samples = int(window_sec * fs)
    total_samples = len(eeg_alpha)
    n_windows = total_samples // window_samples

    alpha_powers = []
    time_minutes = []

    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        segment = eeg_alpha[start:end] * np.hanning(window_samples)

        freqs, psd = signal.welch(segment, fs=fs, window='hann', nperseg=window_samples, noverlap=0, scaling='density')
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        alpha_power = simps(psd[alpha_mask], freqs[alpha_mask])
        alpha_powers.append(alpha_power)
        time_minutes.append(i * window_sec / 60)

    return np.array(time_minutes), np.array(alpha_powers)

def compute_eye_state_vector_events(csv_path):
    MIN_EYE_CLOSED_FRAMES = 90
    df = pd.read_csv(csv_path)
    eye_states = df['Eye State'].values
    timestamps = df['Timestamp (s)'].values

    event_indices = []
    i = 0
    n = len(eye_states)

    while i < n:
        if eye_states[i] == 'CLOSED':
            start = i
            while i < n and eye_states[i] == 'CLOSED':
                i += 1
            closed_duration = i - start
            if closed_duration >= MIN_EYE_CLOSED_FRAMES:
                j = start
                while (j + 1) < n:
                    if eye_states[j] == 'OPEN' and eye_states[j+1] == 'OPEN':
                        break
                    else:
                        closed_duration += 1
                    j += 1
                event_indices.append((start, j))
                i = j
        else:
            i += 1

    return event_indices, timestamps

def compute_all_eye_events(eye_states, timestamps, closed_events, max_time):
    all_events = []
    n = len(eye_states)
    last_idx = 0

    for start, end in closed_events:
        if timestamps[start] > timestamps[last_idx]:
            all_events.append((timestamps[last_idx], timestamps[start], 'OPEN'))
        all_events.append((timestamps[start], timestamps[end], 'CLOSED'))
        last_idx = end

    if timestamps[last_idx] < max_time:
        all_events.append((timestamps[last_idx], max_time, 'OPEN'))

    return all_events

def compute_avg_alpha_for_events(alpha_powers, window_sec, all_events):
    power_fs = 1 / window_sec
    results = []

    for start_time, end_time, label in all_events:
        idx_start = int(start_time * power_fs)
        idx_end = int(end_time * power_fs)
        if idx_start >= len(alpha_powers):
            continue
        if idx_end > len(alpha_powers):
            idx_end = len(alpha_powers)
        segment = alpha_powers[idx_start:idx_end]
        if len(segment) == 0 or np.all(np.isnan(segment)):
            continue
        avg_power = np.nanmean(segment)
        results.append((label, start_time, end_time, avg_power))

    return results

def plot_alpha_power(time_minutes, alpha_powers, smoothed_power=None, epoch_results=None):
    plt.figure(figsize=(12, 6))
    plt.plot(time_minutes, alpha_powers, color='green', alpha=0.5, label='Alpha Power (raw)')
    if smoothed_power is not None:
        plt.plot(time_minutes, smoothed_power, color='red', linewidth=2, label='Alpha Power (smoothed)')

    if epoch_results is not None:
        for i, (_, start, end, power) in enumerate(epoch_results):
            t_min = start / 60
            t_max = end / 60
            plt.hlines(power, t_min, t_max, colors='blue', linestyles='dotted', linewidth=3, label='Epoch Avg' if i == 0 else None)

    plt.xlabel('Time (minutes)')
    plt.ylabel('Alpha Power (V²)')
    plt.yscale('log')
    plt.title('Alpha Power Over Time (8–12 Hz)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    channel = 1
    ADCres = 12
    fsBionode = 5537
    window_sec = 1
    blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
    eye_csv = r"\Users\maryz\EEG-Video\eye_state_log.csv"

    raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
    duration_sec = print_data_stats(len(raw_channel_data), fsBionode)

    eeg_alpha = bandpass_filter_alpha(raw_channel_data, fsBionode)
    time_minutes, alpha_powers = compute_alpha_power(eeg_alpha, fsBionode, window_sec)

    smoothed_power = smooth_alpha_power(alpha_powers, fsBionode, window_sec)

    closed_events, timestamps = compute_eye_state_vector_events(eye_csv)
    df_eye = pd.read_csv(eye_csv)
    all_events = compute_all_eye_events(df_eye['Eye State'].values, df_eye['Timestamp (s)'].values, closed_events, duration_sec)
    results = compute_avg_alpha_for_events(alpha_powers, window_sec, all_events)

    plot_alpha_power(time_minutes, alpha_powers, smoothed_power, epoch_results=results)

    print("\nEpoch Avg Alpha Powers:")
    for i, (label, start, end, power) in enumerate(results):
        print(f"  Epoch {i+1}: {label} ({start:.1f}s - {end:.1f}s) = {power:.4f} V²")

if __name__ == "__main__":
    main()