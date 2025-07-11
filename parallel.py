"""
EEG-Video Alpha Power Analysis Pipeline

Processes EEG data from Bionode or TDT recordings and aligns it with eye-tracking data 
to analyze alpha band (8–12 Hz) power during open and closed eye periods.

Key functionalities:
- Load and scale Neuropulse bin files or TDT data
- Bandpass filter and segment EEG for alpha power extraction
- Match EEG power epochs to eye state intervals
- Visualize raw and smoothed alpha power over time (static)
- Compare average alpha power during OPEN vs. CLOSED eye states

Dependencies: numpy, scipy, pandas, matplotlib, tdt, bionodebinopen
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import simpson as simps
import pandas as pd
from bionodebinopen import fn_BionodeBinOpen
from tdt import read_block

def load_and_preprocess_data(block_path, adc_resolution, fs, channel):
    """
    Loads raw EEG data from a binary file, converts ADC values to voltage,
    handles NaNs, and extracts a specific channel.

    Parameters:
        block_path (str): Path to EEG binary file
        adc_resolution (int): ADC bit resolution (e.g., 12-bit)
        fs (int): Sampling frequency in Hz
        channel (int): Channel index to extract

    Returns:
        np.ndarray: Preprocessed EEG data for selected channel
    """
    data_dict = fn_BionodeBinOpen(block_path, adc_resolution, fs)
    raw_data = np.array(data_dict['channelsData'])
    scale_factor = 1.8 / 4096.0
    raw_data = (raw_data - 2048) * scale_factor
    raw_data = np.nan_to_num(raw_data)
    return raw_data[channel]






def print_data_stats(total_samples, fs):
    """
    Prints basic stats about EEG recording.

    Parameters:
        total_samples (int): Number of samples in the recording
        fs (int): Sampling frequency

    Returns:
        float: Duration of the recording in seconds
    """
    duration_sec = total_samples / fs
    print(f"Total samples: {total_samples}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Total duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")
    return duration_sec





def bandpass_filter_alpha(data, fs):
    """
    Applies a bandpass filter (0.1–25 Hz) to EEG data.

    Parameters:
        data (np.ndarray): Raw EEG signal
        fs (int): Sampling frequency

    Returns:
        np.ndarray: Bandpass-filtered EEG signal
    """
    sos_bandpass = signal.butter(4, [0.1, 25], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos_bandpass, data)




def smooth_alpha_power(alpha_powers, window_sec):
    """
    Applies low-pass filter to smooth alpha power sequence.

    Parameters:
        alpha_powers (np.ndarray): Time series of alpha power values
        fs (int): Original EEG sampling rate
        window_sec (float): Duration used for windowing alpha power (used for resample rate)

    Returns:
        np.ndarray: Smoothed alpha power
    """
    power_fs = 1 / window_sec
    sos_low = signal.butter(4, 0.025, btype='lowpass', fs=power_fs, output='sos')
    return signal.sosfiltfilt(sos_low, alpha_powers)





def compute_alpha_power(eeg_alpha, fs, window_sec):
    """
    Computes average alpha band (8–12 Hz) power per window.

    Parameters:
        eeg_alpha (np.ndarray): EEG signal filtered to alpha band
        fs (int): Sampling frequency
        window_sec (float): Length of each analysis window (sec)

    Returns:
        Tuple[np.ndarray, np.ndarray]: time in minutes, alpha power values
    """
    # Before any segmentation or power calc
    print(f"[DEBUG] Received eeg_alpha len={len(eeg_alpha)}, fs={fs}, window=1 sec")

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
    """
    Detects prolonged eye-closed periods in gaze CSV file.

    Parameters:
        csv_path (str): Path to CSV file containing 'Eye State' and 'Timestamp (s)'

    Returns:
        Tuple[List[Tuple[int, int]], np.ndarray]: closed event index pairs, timestamps
    """
    MIN_EYE_CLOSED_FRAMES = 90
    df = pd.read_csv(csv_path)
    eye_states = df['Eye State'].values
    timestamps = df['Timestamp (s)'].values

    event_indices = []
    i = 0
    n = len(eye_states)

    while i < n:
        if eye_states[i] == 'CLOSED':
            start = i # Mark the start of a closed-eye period
            # Continue while the eyes remain closed
            while i < n and eye_states[i] == 'CLOSED':
                i += 1
            closed_duration = i - start  # Calculate how long eyes stayed closed
            if closed_duration >= MIN_EYE_CLOSED_FRAMES: # if the closed period is long enough
                j = start # Start a secondary index to look for reopening
                while (j + 1) < n: # Extend j forward until two consecutive 'OPEN' states are found
                    if eye_states[j] == 'OPEN' and eye_states[j+1] == 'OPEN':
                        break # Found two opens in a row, end of closed period
                    else:
                        closed_duration += 1 # Extend closed duration if not yet open
                    j += 1
                event_indices.append((start, j)) # Save closed event span
                i = j # Jump to end of event for next scan
        else:
            i += 1 # Skip non-closed frames

    return event_indices, timestamps





def compute_all_eye_events(eye_states, timestamps, closed_events, max_time):
    """
    Builds full OPEN/CLOSED time intervals based on eye vector and closed events.

    Parameters:
        eye_states (np.ndarray): 'OPEN'/'CLOSED' labels
        timestamps (np.ndarray): Corresponding timestamps
        closed_events (List[Tuple[int, int]]): Detected closed intervals
        max_time (float): Final timestamp of EEG or video data

    Returns:
        List[Tuple[float, float, str]]: List of (start, end, label)
    """
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
    """
    Computes average alpha power within each eye event (OPEN/CLOSED).

    Parameters:
        alpha_powers (np.ndarray): Array of alpha power values
        window_sec (float): Time duration of each alpha power window
        all_events (List[Tuple[float, float, str]]): (start, end, label) event tuples

    Returns:
        List[Tuple[str, float, float, float]]: (label, start, end, avg_power)
    """
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
    """
    Plots raw/smoothed alpha power over time, with eye state epochs.

    Parameters:
        time_minutes (np.ndarray): Timestamps for power values in minutes
        alpha_powers (np.ndarray): Raw alpha power values
        smoothed_power (np.ndarray, optional): Smoothed alpha power
        epoch_results (List[Tuple], optional): (label, start, end, avg_power) to overlay
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time_minutes, alpha_powers, color="#5484bc", linewidth=2, label='Alpha Power (raw)')
    if smoothed_power is not None:
        plt.plot(time_minutes, smoothed_power, color='green', linewidth=3, label='Alpha Power (smoothed)')

    if epoch_results is not None:
        for i, (_, start, end, power) in enumerate(epoch_results):
            t_min = start / 60
            t_max = end / 60
            plt.hlines(power, t_min, t_max, colors="#ffa200", linestyles='dotted', linewidth=5.5, label='Epoch Avg' if i == 0 else None)

    plt.xlabel('Time (minutes)')
    plt.ylabel('Alpha Power (V²)')
    plt.yscale('log')
    plt.title('Alpha Power Over Time (8–12 Hz)')
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()





def neuropulse_main():
    channel = 1
    ADCres = 12
    fsBionode = 5537
    window_sec = 1
    blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
    eye_csv = r"\Users\maryz\EEG-Video\esl_neuropulse.csv"

    raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
    duration_sec = print_data_stats(len(raw_channel_data), fsBionode)

    eeg_alpha = bandpass_filter_alpha(raw_channel_data, fsBionode)
    time_minutes, alpha_powers = compute_alpha_power(eeg_alpha, fsBionode, window_sec)

    smoothed_power = smooth_alpha_power(alpha_powers, window_sec)

    closed_events, timestamps = compute_eye_state_vector_events(eye_csv)
    df_eye = pd.read_csv(eye_csv)
    all_events = compute_all_eye_events(df_eye['Eye State'].values, df_eye['Timestamp (s)'].values, closed_events, duration_sec)
    results = compute_avg_alpha_for_events(alpha_powers, window_sec, all_events)

    plot_alpha_power(time_minutes, alpha_powers, smoothed_power, epoch_results=results)

    print("\nEpoch Avg Alpha Powers:")
    for i, (label, start, end, power) in enumerate(results):
        print(f"  Epoch {i+1}: {label} ({start:.1f}s - {end:.1f}s) = {power:.4f} V²")
    
    
def tdt_main():
    channel = 0
    ADCres = 12
    fsBionode = 12207
    window_sec = 2
    blockPath = r"\Users\maryz\EEG-Video\SubjectG-250331-160838"
    eye_csv = r"\Users\maryz\EEG-Video\esl_4.08.csv"

    # raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
    data = read_block(blockPath)
    raw = data.streams.EEGw.data
    # Convert to float32 and scale to volts
    rawC = raw[1:].astype(np.float32)
    # rawC = (rawC - 2048) * 1.8 / (2**ADCres * 1000)  # Scale to volts
    rawC = np.nan_to_num(rawC)
    rawCha = rawC[channel]
    duration_sec = print_data_stats(len(rawCha), fsBionode)

    eeg_alpha = bandpass_filter_alpha(rawCha, fsBionode)
    time_minutes, alpha_powers = compute_alpha_power(eeg_alpha, fsBionode, window_sec)

    smoothed_power = smooth_alpha_power(alpha_powers, fsBionode, window_sec)

    closed_events, timestamps = compute_eye_state_vector_events(eye_csv)
    df_eye = pd.read_csv(eye_csv)
    all_events = compute_all_eye_events(df_eye['Eye State'].values, df_eye['Timestamp (s)'].values, closed_events, duration_sec)
    results = compute_avg_alpha_for_events(alpha_powers, window_sec, all_events)

    plot_alpha_power(time_minutes, alpha_powers, smoothed_power, epoch_results=results)

    print("\nEpoch Avg Alpha Powers:")
    for i, (label, start, end, power) in enumerate(results):
        print(f"  Epoch {i+1}: {label} ({start:.1f}s - {end:.1f}s) = {power:.4e} V²")






if __name__ == "__main__":
    neuropulse_main()
