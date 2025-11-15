"""
EEG-Video Sync & Visualization Script


Synchronizes EEG alpha power data with a gaze-tracked video using a shared timeline.
It visualizes:
- The raw and smoothed alpha power (8–12 Hz) over time
- Filtered EEG signals with annotated eye movement artifacts
- Gaze-tracked video with open/closed eye detection


Dependencies: numpy, scipy, cv2, queue, matplotlib, bionodebinopen
- `parallel.py`: for alpha power preprocessing pipeline
- `gaze_track.py`: for MediaPipe-based eye state detection
- `bionodebinopen.py`: for decoding raw EEG .bin files


Key functionality:
- Load EEG and video from specified paths
- Animate over time with matplotlib: EEG on the right, alpha on bottom, video on top-left
- Eye movements detected in EEG are overlaid in real time
- Press the spacebar to pause/resume playback
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
from bionodebinopen import fn_BionodeBinOpen
import cv2
from threading import Thread
from queue import Queue
from gaze_track import MediaPipeGazeTracking
from parallel import (
    load_and_preprocess_data,
    print_data_stats,
    bandpass_filter_alpha,
    compute_alpha_power,
    smooth_alpha_power
)


# === CONFIG ===
filename = blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
ADCres = 12
fsBionode = 5538
channel = 1
window_sec = 20
step_sec = 0.02
video_path = "video_recordings/alessandro_edit.mp4"


# === Pause flag ===
paused = [False]
video_frame_time = [0.0]  # Used as unified timeline anchor






# === Load and preprocess EEG ===
data = fn_BionodeBinOpen(filename, ADCres, fsBionode)
rawCha = data["channelsData"].astype(np.float32)
raw_data = (rawCha - 2048) * (1.8 / 4096.0)
raw_data = np.nan_to_num(raw_data)




# Convert raw ADC values to voltages (based on 12-bit resolution and 1.8V range)
rawCha = (rawCha - 2**11) * 1.8 / (2**12 * 1000)


# Apply low-pass filter to remove high-frequency noise (e.g., >60 Hz)
b, a = butter(4, 60 / (fsBionode / 2), btype='low')
filtered = filtfilt(b, a, rawCha[channel]) # Clean EEG signal
time = np.arange(len(filtered)) / fsBionode # Time vector in seconds


# === Alpha Power Computation ===
# Compute alpha-band power (8–12 Hz) using parallel pipeline
# raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
raw_channel_data = raw_data[channel]
duration_sec = print_data_stats(len(raw_channel_data), fsBionode)
eeg_alpha = bandpass_filter_alpha(raw_channel_data, fsBionode)
time_min, alpha_power = compute_alpha_power(eeg_alpha, fsBionode, 1)
smoothed_power = smooth_alpha_power(alpha_power, fsBionode, 1)
time_sec_alpha = time_min * 60 # Convert time vector to seconds


# === Video frame queue ===
queue_frame = Queue(maxsize=1)


# === Launch video processing in separate thread ===
def run_video():
    """Launches a video capture thread with gaze tracking (None).


    Captures video from a given path, processes each frame to detect gaze and blinking,
    and places annotated frames in a queue for visualization. Also syncs video time with EEG display.
    """
    cap = cv2.VideoCapture(video_path)
    gaze = MediaPipeGazeTracking()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    while cap.isOpened():
        if paused[0]:
            cv2.waitKey(1)
            continue
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (960, 720)) # Resize for consistent display
        current_time = frame_count / fps
        video_frame_time[0] = current_time   # Global time sync point
        frame_count += 1
       
        gaze.refresh(frame)  # Run face/gaze detection
        gaze.is_blinking(current_time)
        annotated = gaze.annotated_frame(current_time) # Annotate blinking
       
        # Update video frame queue for animation
        if queue_frame.empty():
            queue_frame.put(annotated)
        cv2.waitKey(1) # Prevents GUI freeze


    cap.release()
    # gaze.export_to_csv() # Save tracking data to file


video_thread = Thread(target=run_video, daemon=True)
video_thread.start()


# === Plotting ===
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 2)
ax_video = fig.add_subplot(gs[0, 0])
ax_eeg = fig.add_subplot(gs[0, 1])
ax_alpha = fig.add_subplot(gs[1, :])


video_image = ax_video.imshow(np.zeros((480, 853, 3), dtype=np.uint8))
ax_video.axis('off')
ax_video.set_title("Video Feed (Gaze Tracked)")


line, = ax_eeg.plot([], [], color='blue')
event_dots, = ax_eeg.plot([], [], 'ro')
ax_eeg.set_xlabel("Time (s)")
ax_eeg.set_ylabel("Voltage (V)")
ax_eeg.set_title("Filtered EEG with Eye Movement Detection")
# ax_eeg.grid(True)
ax_eeg.set_xlim(0, window_sec)
ax_eeg.set_ylim(-0.00007, 0.00007)


raw_line, = ax_alpha.plot([], [], label='Raw Alpha Power (V²)', color='green', alpha=0.9)
smooth_line, = ax_alpha.plot([], [], label='Smoothed Alpha Power', color='red', linewidth=2)
ax_alpha.set_xlabel('Time (s)')
ax_alpha.set_ylabel('Alpha Power (V²)')
ax_alpha.set_yscale('log')
ax_alpha.set_title('Animated Alpha Power (20s Window)')
# ax_alpha.grid(True)
ax_alpha.legend()


text_labels = []


# Detects rapid eye movement spikes in EEG window (list of tuples).
def detect_eye_movements(y_win, t_win):
    """Detects rapid eye movement spikes in EEG window (list of tuples).


    y_win: np.ndarray
        EEG signal values in a moving window.
    t_win: np.ndarray
        Corresponding timestamps for y_win samples.


    Returns:
        events: list of tuples
            List of (time, voltage) pairs representing detected eye movement events.
    """
    threshold_spike = 0.0005 # Voltage threshold for upward spike
    threshold_dip = -0.0001 # Follow-up negative dip threshold
    max_gap_sec = 0.12 # Max time between spike/dip
    max_gap_samples = int(max_gap_sec * fsBionode)
    events = []
    i = 0
    while i < len(y_win) - max_gap_samples:
        if y_win[i] > threshold_spike:
            for j in range(i + 1, min(i + max_gap_samples, len(y_win))):
                if y_win[j] < threshold_dip:
                    mid_idx = i + (j - i) // 2
                    events.append((t_win[mid_idx], y_win[mid_idx]))
                    i = j + 1
                    break
            else:
                i += 1
        else:
            i += 1
    return events


def init():
    """Initializes empty plots for animation setup (list).


    Clears and resets all animation objects: EEG line, event dots, video image, alpha power lines.


    No parameters.
    Returns:
        list
            Updated plot elements to initialize the animation.
    """
    line.set_data([], [])
    event_dots.set_data([], [])
    video_image.set_array(np.zeros((480, 853, 3), dtype=np.uint8))
    raw_line.set_data([], [])
    smooth_line.set_data([], [])
    ax_alpha.set_xlim(0, 20)
    ax_alpha.set_ylim(np.min(alpha_power), np.max(alpha_power))
    return [line, event_dots, video_image, raw_line, smooth_line]


def update(_):
    """Animation update function that refreshes plots with current video and EEG data (list).


    _: int
        Frame index passed by matplotlib animation system (unused).


    Returns:
        list
            Updated plot elements for EEG, alpha power, and video frame.
    """
    global text_labels
    if paused[0]:
        return [line, event_dots, video_image, raw_line, smooth_line] + text_labels


    current_time = video_frame_time[0]


    # === EEG update ===
    eeg_mask = (time >= current_time) & (time <= current_time + window_sec)
    if np.any(eeg_mask):
        t_win = time[eeg_mask]
        y_win = filtered[eeg_mask]
        if len(t_win) == 0 or len(y_win) == 0:
            return [line, event_dots, video_image, raw_line, smooth_line] + text_labels
        t_win_relative = t_win - t_win[0]
        line.set_data(t_win_relative, y_win)
        ax_eeg.set_title(f"Filtered EEG ({t_win[0]:.1f}s - {t_win[-1]:.1f}s)")


        # Detect and plot eye movement events
        events = detect_eye_movements(y_win, t_win)
        t_events, y_events = [], []
        if events:
            abs_t_events, y_events = zip(*events)
            t_events = [t - t_win[0] for t in abs_t_events]
            event_dots.set_data(t_events, y_events)
        else:
            event_dots.set_data([], [])
        for txt in text_labels:
            if txt in ax_eeg.texts:
                txt.remove()
        text_labels = []
        for tx, ty in zip(t_events, y_events):
            txt = ax_eeg.text(tx, ty + 0.00004, 'Eye Movement/EOG', color='red', fontsize=8)
            text_labels.append(txt)


    # === Alpha power update ===
    t_start = max(current_time - 20, 0)
    t_end = current_time
    try:
        alpha_mask = (time_sec_alpha >= current_time) & (time_sec_alpha <= current_time + window_sec)
        if np.any(alpha_mask):
            raw_line.set_data(time_sec_alpha[alpha_mask] - current_time, alpha_power[alpha_mask])
            smooth_line.set_data(time_sec_alpha[alpha_mask] - current_time, smoothed_power[alpha_mask])
            ax_alpha.set_xlim(0, window_sec)
    except Exception as e:
        print(f"Alpha plot error at time {current_time:.2f}s: {e}")


    # === Video ===
    if not queue_frame.empty():
        frame = queue_frame.get()
        video_image.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    try:
        print(f"Video Time: {current_time:.2f}s | EEG Window: {t_win[0]:.2f}s | Alpha Power Window: {t_start:.2f}-{t_end:.2f}s")
    except Exception as e:
        print(f"[DEBUG] Print failed at {current_time:.2f}s: {e}")


    # print(f"Video Time: {current_time:.2f}s | EEG Window: {t_win[0]:.2f}s | Alpha Power Window: {t_start:.2f}-{t_end:.2f}s")


    return [line, event_dots, video_image, raw_line, smooth_line] + text_labels


def on_key(event):
    """Handles keypress events to pause/resume the animation (None).


    event: matplotlib.backend_bases.KeyEvent
        The keyboard event triggered by user input.


    Returns:
        None
    """
    if event.key == ' ':
        paused[0] = not paused[0]
        print("Paused" if paused[0] else "Resumed")


fig.canvas.mpl_connect('key_press_event', on_key)


ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=step_sec * 1000,
    blit=True,
    cache_frame_data=False
)


plt.tight_layout()
plt.show()



