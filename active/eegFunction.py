import numpy as np
import pandas as pd
import cv2
from threading import Thread
from queue import Queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt, sosfiltfilt, sosfilt
from scipy import signal
from scipy.integrate import simpson as simps
from active.bionodebinopen import fn_BionodeBinOpen
from active.gaze_track import MediaPipeGazeTracking
from tdt import read_block

def eegDetect(block_path, eye_csv, video_path=None, channel=1, adc_res=12, fs=5537, window_sec_alpha=1.0, window_sec_eeg=20):
    """
    Unified EEG + Video + Eye State Analysis Function
    Combines Neuropulse/TDT EEG loading, alpha power computation, gaze/eye tracking, and animated visualization.

    Parameters:
        block_path (str): Path to EEG .bin or TDT block
        eye_csv (str): CSV containing 'Eye State' and 'Timestamp (s)'
        video_path (str, optional): Path to gaze-tracked video (if visualization desired)
        channel (int): EEG channel index
        adc_res (int): ADC resolution
        fs (int): EEG sampling frequency
        window_sec_alpha (float): Window length for alpha power (seconds)
        window_sec_eeg (float): Moving window length for EEG plot (seconds)
    """

    # --- Load EEG ---
    if block_path.endswith(".bin"):
        data_dict = fn_BionodeBinOpen(block_path, adc_res, fs)
        raw = np.array(data_dict['channelsData'], dtype=np.float32)
        raw = (raw - 2048) * (1.8 / 4096.0)
        raw = np.nan_to_num(raw)
        raw_chan = raw[channel]
    else:
        data = read_block(block_path)
        raw = data.streams.EEGw.data.astype(np.float32)
        raw = np.nan_to_num(raw)
        raw_chan = raw[channel]

    total_samples = len(raw_chan)
    duration_sec = total_samples / fs
    print(f"EEG samples: {total_samples}, duration: {duration_sec:.2f}s")

    # --- Bandpass Alpha ---
    sos_alpha = butter(4, [8, 12], btype='band', fs=fs, output='sos')
    eeg_alpha = sosfilt(sos_alpha, raw_chan)

    # --- Alpha Power Computation ---
    window_samples_alpha = int(window_sec_alpha * fs)
    n_windows = len(eeg_alpha) // window_samples_alpha
    alpha_powers = []
    time_alpha_sec = []
    for i in range(n_windows):
        seg = eeg_alpha[i*window_samples_alpha:(i+1)*window_samples_alpha] * np.hanning(window_samples_alpha)
        freqs, psd = signal.welch(seg, fs=fs, window='hann', nperseg=window_samples_alpha, noverlap=0, scaling='density')
        mask = (freqs >= 8) & (freqs <= 12)
        alpha_powers.append(simps(psd[mask], freqs[mask]))
        time_alpha_sec.append(i * window_sec_alpha)
    alpha_powers = np.array(alpha_powers)
    time_alpha_sec = np.array(time_alpha_sec)
    smoothed_power = sosfiltfilt(butter(4, 0.025, btype='low', fs=1/window_sec_alpha, output='sos'), alpha_powers)

    # --- Eye State Events ---
    df_eye = pd.read_csv(eye_csv)
    eye_states = df_eye['Eye State'].values
    timestamps = df_eye['Timestamp (s)'].values

    # Find closed-eye events
    MIN_CLOSED = 90
    closed_events = []
    i = 0
    while i < len(eye_states):
        if eye_states[i] == 'CLOSED':
            start = i
            while i < len(eye_states) and eye_states[i] == 'CLOSED':
                i += 1
            if i - start >= MIN_CLOSED:
                closed_events.append((start, i-1))
        else:
            i += 1

    # Build full open/closed intervals
    all_events = []
    last_idx = 0
    for start, end in closed_events:
        if timestamps[start] > timestamps[last_idx]:
            all_events.append((timestamps[last_idx], timestamps[start], 'OPEN'))
        all_events.append((timestamps[start], timestamps[end], 'CLOSED'))
        last_idx = end
    if timestamps[last_idx] < duration_sec:
        all_events.append((timestamps[last_idx], duration_sec, 'OPEN'))

    # --- Compute Avg Alpha per Event ---
    results = []
    alpha_fs = 1 / window_sec_alpha
    for start, end, label in all_events:
        idx_start = int(start * alpha_fs)
        idx_end = int(end * alpha_fs)
        idx_end = min(idx_end, len(alpha_powers))
        segment = alpha_powers[idx_start:idx_end]
        if len(segment) == 0:
            continue
        results.append((label, start, end, np.nanmean(segment)))

    # --- Video Thread + Queue ---
    paused = [False]
    video_frame_time = [0.0]
    queue_frame = Queue(maxsize=1)
    def run_video():
        if video_path is None:
            return
        cap = cv2.VideoCapture(video_path)
        gaze = MediaPipeGazeTracking()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        while cap.isOpened():
            if paused[0]:
                cv2.waitKey(1)
                continue
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (960,720))
            current_time = frame_idx / fps
            video_frame_time[0] = current_time
            frame_idx += 1
            gaze.refresh(frame)
            gaze.is_blinking(current_time)
            annotated = gaze.annotated_frame(current_time)
            if queue_frame.empty():
                queue_frame.put(annotated)
            cv2.waitKey(1)
        cap.release()
    if video_path:
        Thread(target=run_video, daemon=True).start()

    # --- Eye Movements Detection in EEG ---
    def detect_eye_movements(y_win, t_win):
        threshold_spike = 0.0005
        threshold_dip = -0.0001
        max_gap_samples = int(0.12 * fs)
        events = []
        i = 0
        while i < len(y_win) - max_gap_samples:
            if y_win[i] > threshold_spike:
                for j in range(i+1, min(i+max_gap_samples, len(y_win))):
                    if y_win[j] < threshold_dip:
                        mid = i + (j-i)//2
                        events.append((t_win[mid], y_win[mid]))
                        i = j + 1
                        break
                else:
                    i += 1
            else:
                i += 1
        return events

    # --- Matplotlib Animation ---
    fig = plt.figure(figsize=(16,8))
    gs = fig.add_gridspec(2,2)
    ax_video = fig.add_subplot(gs[0,0])
    ax_eeg = fig.add_subplot(gs[0,1])
    ax_alpha = fig.add_subplot(gs[1,:])
    video_image = ax_video.imshow(np.zeros((480,853,3),dtype=np.uint8))
    ax_video.axis('off')
    line, = ax_eeg.plot([],[],color='blue')
    event_dots, = ax_eeg.plot([],[], 'ro')
    ax_eeg.set_xlim(0, window_sec_eeg)
    ax_eeg.set_ylim(np.min(raw_chan), np.max(raw_chan))
    raw_line, = ax_alpha.plot([],[],label='Raw Alpha', color='green')
    smooth_line, = ax_alpha.plot([],[],label='Smoothed Alpha', color='red', linewidth=2)
    ax_alpha.set_xlabel('Time (s)')
    ax_alpha.set_ylabel('Alpha Power')
    ax_alpha.set_yscale('log')
    ax_alpha.legend()
    text_labels = []

    def init():
        line.set_data([],[])
        event_dots.set_data([],[])
        video_image.set_array(np.zeros((480,853,3),dtype=np.uint8))
        raw_line.set_data([],[])
        smooth_line.set_data([],[])
        return [line, event_dots, video_image, raw_line, smooth_line]

    def update(_):
        nonlocal text_labels
        if paused[0]:
            return [line, event_dots, video_image, raw_line, smooth_line] + text_labels
        current_time = video_frame_time[0]
        # EEG window
        mask = (np.arange(len(raw_chan))/fs >= current_time) & (np.arange(len(raw_chan))/fs <= current_time+window_sec_eeg)
        if np.any(mask):
            t_win = np.arange(len(raw_chan))/fs
            y_win = raw_chan
            line.set_data(t_win[mask]-t_win[mask][0], y_win[mask])
            events = detect_eye_movements(y_win[mask], t_win[mask])
            if events:
                abs_t, y_ev = zip(*events)
                t_ev = [t - t_win[mask][0] for t in abs_t]
                event_dots.set_data(t_ev, y_ev)
            else:
                event_dots.set_data([],[])
            for txt in text_labels:
                if txt in ax_eeg.texts:
                    txt.remove()
            text_labels.clear()
            for tx, ty in zip(*zip(*events)) if events else ([],[]):
                txt = ax_eeg.text(tx, ty+0.00004, 'Eye Movement/EOG', color='red', fontsize=8)
                text_labels.append(txt)
        # Alpha plot
        mask_alpha = (time_alpha_sec >= current_time) & (time_alpha_sec <= current_time+window_sec_eeg)
        if np.any(mask_alpha):
            raw_line.set_data(time_alpha_sec[mask_alpha]-current_time, alpha_powers[mask_alpha])
            smooth_line.set_data(time_alpha_sec[mask_alpha]-current_time, smoothed_power[mask_alpha])
        # Video
        if video_path and not queue_frame.empty():
            frame = queue_frame.get()
            video_image.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return [line, event_dots, video_image, raw_line, smooth_line] + text_labels

    def on_key(event):
        if event.key==' ':
            paused[0] = not paused[0]
            print("Paused" if paused[0] else "Resumed")
    fig.canvas.mpl_connect('key_press_event', on_key)

    if video_path:
        ani = animation.FuncAnimation(fig, update, init_func=init, interval=20, blit=True)
        plt.show()

    # --- Print Event Alpha Summary ---
    print("\nEpoch Avg Alpha Powers:")
    for i, (label, start, end, power) in enumerate(results):
        print(f"  Epoch {i+1}: {label} ({start:.1f}-{end:.1f}s) = {power:.4e} V²")
    return results

