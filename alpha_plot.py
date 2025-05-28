import warnings
# Suppress protobuf deprecation warning from MediaPipe
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated.*",
    category=UserWarning
)
import numpy as np
import pandas as pd
 # Track previous blink status and last blink end time for post-blink exclusion
prev_blink_closed = False  # Track previous blink status
last_blink_end_time = -np.inf  # Timestamp when last blink ended
import scipy.signal as signal
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bionodebinopen import fn_BionodeBinOpen
import cv2
import mediapipe as mp
import time
from threading import Thread
from queue import Queue
from collections import deque
from gaze_track import MediaPipeGazeTracking
from scipy.signal import resample_poly

# EEG CONFIG
channel = 2
fsBionode = 5282  # 12.5 kHz
fs_proc = 1000  # new sampling rate after downsampling
ADCres = 12
window_sec = 10  # 10 seconds window for raw EEG signal
window_size = 0.2
# step_sec = 0.02
# Define separate sample windows for raw and alpha segments
raw_window_samples = int(window_sec * fs_proc)
alpha_window_samples = int(window_size * fs_proc)
# Original-sampling parameters for alpha power calculation
alpha_window_samples_orig = int(window_size * fsBionode)
blockPath = '/Users/alessandroascaniorsini/Documents/GitHub/EEG-Video-main/bin_files/ear3.31.25_1.bin'

# VIDEO CONFIG
video_path = "video_recordings/alessandro.mov"

# Determine video FPS and sync plotting interval
cap_temp = cv2.VideoCapture(video_path)
fps = cap_temp.get(cv2.CAP_PROP_FPS)
cap_temp.release()
frame_interval = 1.0 / fps  # seconds per video frame
step_sec = frame_interval
# Original-sampling step size for alpha power calculation
step_samples_orig = int(step_sec * fsBionode)
step_samples = int(step_sec * fs_proc)

# SIGNAL START DELAY CONFIGURATION
signal_delay_sec = 0.0  # seconds to delay signal start; adjust as needed
delay_samples = int(signal_delay_sec * fs_proc)

# EEG PROCESSING
def preprocess_eeg():
    rawCha = np.array(fn_BionodeBinOpen(blockPath, ADCres, fsBionode)['channelsData'])
    rawCha = (rawCha - 2**11) * 1.8 / (2**12)
    rawCha = np.nan_to_num(rawCha, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply a 4th-order Butterworth band-pass filter from 7 to 13 Hz
    sos_bp = signal.butter(4, [7, 13], btype='bandpass', fs=fsBionode, output='sos')
    eeg_filtered = signal.sosfiltfilt(sos_bp, rawCha[channel - 1])
    # Downsample
    eeg_proc = resample_poly(eeg_filtered, up=int(fs_proc), down=int(fsBionode))
    return eeg_filtered, eeg_proc

# Shared data
queue_frame = Queue()
eeg_original, eeg_raw = preprocess_eeg()

# === Offline computation of eyes-open/closed state change times ===
import cv2
from gaze_track import MediaPipeGazeTracking

event_times = []
cap_static = cv2.VideoCapture(video_path)
# Progress reporting for offline event_times computation
total_frames = int(cap_static.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Offline state computation: {total_frames} frames to process")
progress_interval_frames = max(1, total_frames // 20)  # update every 5%
gaze_static = MediaPipeGazeTracking()
fps_vid = cap_static.get(cv2.CAP_PROP_FPS)
prev_status = False
pending_status = None
pending_start = None
frame_idx = 0

while True:
    ret, frame_static = cap_static.read()
    if not ret:
        break
    t = frame_idx / fps_vid
    frame_idx += 1
    if frame_idx % progress_interval_frames == 0 or frame_idx == total_frames:
        pct = frame_idx / total_frames * 100
        print(f"[State] Frame {frame_idx}/{total_frames} ({pct:.1f}%)")

    gaze_static.refresh(frame_static)
    closed = gaze_static.is_blinking(t)

    # Debounce: initiate or handle pending state changes
    if pending_status is None:
        if closed != prev_status:
            pending_status = closed
            pending_start = t
    else:
        # pending_status is not None
        if closed == pending_status:
            if t - pending_start >= 3.0:
                # Commit the state change
                prev_status = pending_status
                event_times.append(t)
                pending_status = None
        else:
            # Reset if state flips before debounce completes
            pending_status = None

cap_static.release()
# === End offline state-change computation ===

# --- Pre-calculated full-session plot for reference (not animated) ---
# Calculate full-session alpha power
full_times = np.arange(0, len(eeg_original) / fsBionode, step_sec)
total_steps = len(full_times)
print(f"Full-session alpha power: {total_steps} steps to compute")
progress_interval_steps = max(1, total_steps // 20)
full_power = []
for i, t in enumerate(full_times):
    idx = int(t * fsBionode)
    seg = eeg_original[idx:idx + alpha_window_samples_orig]
    if len(seg) < alpha_window_samples_orig:
        break
    freqs, psd = signal.welch(
        seg,
        fs=fsBionode,
        window='hann',
        nperseg=alpha_window_samples_orig,
        noverlap=alpha_window_samples_orig // 2,
        scaling='density'
    )
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    full_power.append(np.sum(psd[alpha_mask]))
    if i % progress_interval_steps == 0 or i == total_steps - 1:
        pct = i / total_steps * 100
        print(f"[Alpha] Step {i+1}/{total_steps} ({pct:.1f}%)")

# === Export alpha power with epochs to CSV ===
csv_times = full_times[:len(full_power)]
# Mark epoch events: 1 if a change event occurs at that timestamp (within half a step), else 0
csv_epochs = [1 if any(abs(t - et) <= (step_sec / 2) for et in event_times) else 0 for t in csv_times]
df = pd.DataFrame({
    'time_s': csv_times,
    'alpha_power': full_power,
    'epoch_flag': csv_epochs
})
csv_path = 'alpha_power_epochs.csv'
df.to_csv(csv_path, index=False)
print(f"Saved CSV to {csv_path}")
# === End CSV export ===

# Plot full-session alpha power (for reference)
fig_full, ax_full = plt.subplots(figsize=(12, 4))
#ax_full.plot(full_times[:len(full_power)], full_power)
ax_full.plot(full_times[:len(full_power)], full_power, color='black')
ax_full.set_xlabel('Time (s)')
ax_full.set_ylabel('Alpha power (V²)')
ax_full.set_title('Full-session Alpha Power (8–12 Hz)')
# Add vertical red lines for eyes-open/closed state changes
for et in event_times:
    ax_full.axvline(x=et, color='red', linestyle='--')
plt.tight_layout()
plt.show()


# Shared state for eye-closure detection
# Variables for alpha power averaging per state (eyes open vs closed)
# state_start_idx = 0  # Sample index when current state began
# state_alpha_accum = []  # Accumulate alpha power values for the current state
avg_alpha_open = None  # Average alpha power for eyes-open state
avg_alpha_closed = None  # Average alpha power for eyes-closed state
std_alpha_open = None  # Standard deviation of alpha power for eyes-open state
std_alpha_closed = None  # Standard deviation of alpha power for eyes-closed state

current_eye_closed = False
current_status = False  # Current eye state (True if closed, False if open)
# For tracking eye state change events
pending_eye_closed = None  # Candidate state for debounce
pending_start_idx = None   # Sample index when pending state began
event_states = []
event_lines_raw = []
event_lines_alpha = []

# Buffers for 20-second window of alpha power per eye state
open_alpha_buffer = deque(maxlen=int(30 / step_sec))
closed_alpha_buffer = deque(maxlen=int(30 / step_sec))

 # Pause control
paused = False

# VIDEO THREAD
def run_video():
    global current_eye_closed
    cap = cv2.VideoCapture(video_path)
    gaze = MediaPipeGazeTracking()
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    frame_count = 0

    # Use real-time frame interval to throttle reading
    frame_interval = 1.0 / fps

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 720))
        current_time = frame_count / fps
        frame_count += 1

        # Throttle video reading to match real-time FPS
        elapsed = time.time() - start_time
        to_wait = frame_interval - elapsed
        if to_wait > 0:
            time.sleep(to_wait)

        gaze.refresh(frame)
        # Determine eye state: True if closed, False if open
        closed = gaze.is_blinking(current_time)
        #print(f"Time {current_time}: Eye closed: {closed}");
        current_eye_closed = closed
        annotated = gaze.annotated_frame(current_time)
        queue_frame.put((annotated, current_time))

    cap.release()
    gaze.export_to_csv()

# EEG PLOT: raw signal over window
# step_samples = int(step_sec * fs_proc)

# Figure layout: video + raw EEG + alpha PSD
fig, (ax_video, ax_raw, ax_alpha) = plt.subplots(3, 1, figsize=(12, 12))

# Text overlays for average alpha power
text_open = ax_video.text(
    0.01, 0.1,
    "Avg Open Alpha: N/A",
    transform=ax_video.transAxes,
    color='white',
    fontsize=12,
    backgroundcolor='black',
    animated=True
)
text_closed = ax_video.text(
    0.01, 0.0,
    "Avg Closed Alpha: N/A",
    transform=ax_video.transAxes,
    color='white',
    fontsize=12,
    backgroundcolor='black',
    animated=True
)

# Tracking EEG detection state
detection_start_time = None  # Timestamp when condition first met
detection_detected = False   # Whether EEG has been detected

# Orange text overlay for EEG detection status at top-right
text_detection = ax_video.text(
    0.98, 0.95,
    "EEG not detected",
    transform=ax_video.transAxes,
    color='white',
    backgroundcolor='black',
    fontsize=14,
    ha='right',
    animated=True
)

# Raw EEG signal plot
line_raw, = ax_raw.plot([], [], animated=True)
ax_raw.set_xlabel('Time (s)')
ax_raw.set_ylabel('Amplitude (V)')
ax_raw.set_title('Raw EEG Signal (10 s window)')
ax_raw.set_xlim(-window_sec, 0)

# Indicator for current time on raw signal
vline = ax_raw.axvline(x=0, color='r', linestyle='--', animated=True)

# Alpha power over time plot
line_alpha, = ax_alpha.plot([], [], animated=True)
ax_alpha.set_xlim(-window_sec, 0)
ax_alpha.set_xlabel('Time (s)')
ax_alpha.set_ylabel('Alpha power (V²)')
ax_alpha.set_title('Alpha Power (8–12 Hz) Over Time')
# Display alpha power on a log scale
#ax_alpha.set_yscale('log', nonpositive='clip')
# Set initial y-limit to ensure positive values for log scale
ax_alpha.set_ylim(1e-12, 1e-6)
# Initialize buffer for alpha power (10 s history)
power_buffer = deque(maxlen=int(window_sec / step_sec))
# Buffer for raw EEG signal (10 s history)
raw_buffer = deque(maxlen=raw_window_samples)
# Historical extremes for y-axis limits
raw_min = float('inf')
raw_max = float('-inf')
power_min = float('inf')
power_max = float('-inf')

# Video display
img_disp = ax_video.imshow(np.zeros((720, 960, 3), dtype=np.uint8))
ax_video.axis('off')

# Update function
def update(frame):
    global paused, text_open, text_closed, text_detection, current_status
    global raw_min, raw_max, power_min, power_max
    global avg_alpha_open, avg_alpha_closed, std_alpha_open, std_alpha_closed
    global pending_eye_closed, pending_start_time
    global detection_start_time, detection_detected
    global prev_blink_closed, last_blink_end_time
    if paused:
        return line_raw, vline, img_disp, line_alpha, text_open, text_closed, text_detection
    if queue_frame.empty():
        return line_raw, vline, img_disp, line_alpha, text_open, text_closed, text_detection
    frame, current_time = queue_frame.get()
    img_disp.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if current_time < signal_delay_sec:
        return line_raw, vline, img_disp, line_alpha, text_open, text_closed, text_detection

    idx_eff = int(current_time * fs_proc) - delay_samples

    # Detect blink end to mark post-blink exclusion
    if prev_blink_closed and not current_eye_closed:
        last_blink_end_time = current_time
    prev_blink_closed = current_eye_closed

    # Debounce state changes: only commit after persisting for >=3s
    if current_eye_closed != current_status and pending_eye_closed is None:
        # Start tracking a candidate state change
        pending_eye_closed = current_eye_closed
        print(f"Pending state change to {pending_eye_closed} at {current_time:.2f}s")
        pending_start_time = current_time
    # Check if pending state has persisted long enough
    if pending_eye_closed is not None:
        pending_duration = current_time - pending_start_time
        print(f"Pending state: {pending_eye_closed}, Duration: {pending_duration:.2f}s")
        if pending_duration >= 3.0:
            print(f"Committing state change to {pending_eye_closed} at {current_time:.2f}s")
            # Commit the state change
            current_status = pending_eye_closed
            # record event time and state
            event_times.append(current_time)
            event_states.append(current_status)
            # create vertical lines at x=0 with the appropriate color
            color = 'red' if current_status else 'green'
            line_raw_event = ax_raw.axvline(x=0, color=color, linestyle='--', animated=True)
            line_alpha_event = ax_alpha.axvline(x=0, color=color, linestyle='--', animated=True)
            event_lines_raw.append(line_raw_event)
            event_lines_alpha.append(line_alpha_event)
            # Clear pending
            pending_eye_closed = None
            pending_start_time = None
        else:
            if pending_eye_closed != current_eye_closed:
                # Reset pending state if the eye state changed again before 3s
                print(f"Resetting pending state change at {current_time:.2f}s")
                pending_eye_closed = None
                pending_start_time = None

    if idx_eff + raw_window_samples > len(eeg_raw):
        return line_raw, vline, img_disp, line_alpha

    # Append new samples to the raw buffer
    new_samples = eeg_raw[idx_eff:idx_eff + step_samples]
    raw_buffer.extend(new_samples)

    # Relative time axis for buffered raw signal
    times = np.linspace(-window_sec, 0, len(raw_buffer))

    # Update vertical line for current time
    vline.set_xdata([0, 0])

    # Color the vertical line: red for eyes closed, green for eyes open
    if current_eye_closed:
        vline.set_color('red')
    else:
        vline.set_color('green')

    # Update raw signal plot using the buffer
    line_raw.set_data(times, list(raw_buffer))

    # Update historical extremes for raw EEG signal
    current_min, current_max = min(raw_buffer), max(raw_buffer)
    raw_min = min(raw_min, current_min)
    raw_max = max(raw_max, current_max)
    # Set y-axis limits based on historical extremes
    y_margin = 0.1 * (raw_max - raw_min) if raw_max > raw_min else raw_max * 0.1
    ax_raw.set_ylim(raw_min - y_margin, raw_max + y_margin)

    # Calculate alpha power on the original-sampled signal
    delay_orig = int(signal_delay_sec * fsBionode)
    idx_orig = int(current_time * fsBionode) - delay_orig
    segment_alpha = eeg_original[idx_orig:idx_orig + alpha_window_samples_orig]
    freqs, psd = signal.welch(
        segment_alpha,
        fs=fsBionode,
        window='hann',
        nperseg=alpha_window_samples_orig,
        noverlap=alpha_window_samples_orig // 2,
        scaling='density'
    )
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    alpha_power = np.sum(psd[alpha_mask])
    # Append new alpha power
    power_buffer.append(alpha_power)

    # Only include in averages if not blinking or within 0.2s after blink
    if not (current_time <= last_blink_end_time + 0.4):
        if current_status:  # eyes closed
            closed_alpha_buffer.append(alpha_power)
            avg_alpha_closed = np.mean(closed_alpha_buffer)
            std_alpha_closed = np.std(closed_alpha_buffer)
        else:  # eyes open
            open_alpha_buffer.append(alpha_power)
            avg_alpha_open = np.mean(open_alpha_buffer)
            std_alpha_open = np.std(open_alpha_buffer)

    # Relative time axis for alpha power
    alpha_times = np.linspace(-window_sec+window_size, 0, len(power_buffer))
    line_alpha.set_data(alpha_times, list(power_buffer))
    # Update historical extremes for alpha power
    current_p_min, current_p_max = min(power_buffer), max(power_buffer)
    power_min = min(power_min, current_p_min)
    power_max = max(power_max, current_p_max)
    # Set y-axis limits based on historical extremes
    margin_p = 0.1 * (power_max - power_min) if power_max > power_min else power_max * 0.1
    ax_alpha.set_ylim(power_min - margin_p, power_max + margin_p)

    # Update event lines positions and cleanup old ones
    for i in reversed(range(len(event_times))):
        et = event_times[i]
        x_rel = et - current_time
        if x_rel < -window_sec:
            # remove lines that moved out of the window
            event_lines_raw[i].remove()
            event_lines_alpha[i].remove()
            del event_times[i]
            del event_states[i]
            del event_lines_raw[i]
            del event_lines_alpha[i]
        else:
            event_lines_raw[i].set_xdata([x_rel, x_rel])
            event_lines_alpha[i].set_xdata([x_rel, x_rel])

    # Update text overlays with the latest average alpha values and standard deviations
    if avg_alpha_open is not None and std_alpha_open is not None:
        text_open.set_text(f"Avg Open Alpha: {avg_alpha_open:.6e} ± {std_alpha_open:.6e}")
    if avg_alpha_closed is not None and std_alpha_closed is not None:
        text_closed.set_text(f"Avg Closed Alpha: {avg_alpha_closed:.6e} ± {std_alpha_closed:.6e}")

    # Update EEG detection status: closed alpha > open alpha for >=5s
    global detection_start_time, detection_detected
    if avg_alpha_open is not None and avg_alpha_closed is not None:
        if avg_alpha_closed > avg_alpha_open:
            if detection_start_time is None:
                detection_start_time = current_time
            if current_time - detection_start_time >= 5.0:
                detection_detected = True
        else:
            detection_start_time = None
            detection_detected = False
        # Update detection text
        text_detection.set_text("EEG detected" if detection_detected else "EEG not detected")
        # Update detection text color: white when not detected, orange when detected
        text_detection.set_color('orange' if detection_detected else 'white')

    return line_raw, vline, img_disp, line_alpha, text_open, text_closed, text_detection

# Start video thread
Thread(target=run_video, daemon=True).start()

ani = animation.FuncAnimation(
    fig,
    update,
    init_func=lambda: (line_raw, vline, img_disp, line_alpha, text_open, text_closed, text_detection),
    interval=frame_interval * 1000,
    blit=True,
    cache_frame_data=False
)
#
# Toggle pause on spacebar
def on_key(event):
    global paused, index
    if event.key == ' ':
        paused = not paused
        # Clear queued video frames when pausing to avoid backlog
        if paused:
            with queue_frame.mutex:
                queue_frame.queue.clear()
        print("Paused" if paused else "Resumed")

fig.canvas.mpl_connect('key_press_event', on_key)

plt.tight_layout()
plt.show()
