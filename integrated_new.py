# # eeg_gaze_alpha_plot.py

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from scipy.signal import butter, filtfilt
# from bionodebinopen import fn_BionodeBinOpen
# import cv2
# from threading import Thread
# from queue import Queue
# from gaze_track import MediaPipeGazeTracking
# from parallel import (
#     load_and_preprocess_data,
#     print_data_stats,
#     bandpass_filter_alpha,
#     compute_alpha_power,
#     smooth_alpha_power
# )

# # === CONFIG ===
# filename = blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
# ADCres = 12
# fsBionode = 6250
# channel = 1
# window_sec = 20
# step_sec = 0.02
# video_path = "video_recordings/alessandro_edit.mp4"

# # === Pause flag ===
# paused = [False]
# video_frame_time = [0.0]  # Used as unified timeline anchor

# # === Load and preprocess EEG ===
# data = fn_BionodeBinOpen(filename, ADCres, fsBionode)
# rawCha = data["channelsData"].astype(np.float32)
# rawCha = (rawCha - 2**11) * 1.8 / (2**12 * 1000)
# highCutoff = 60
# b, a = butter(4, highCutoff / (fsBionode / 2), btype='low')
# filtered = filtfilt(b, a, rawCha[channel])
# time = np.arange(len(filtered)) / fsBionode

# # === Alpha Power Computation ===
# raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
# duration_sec = print_data_stats(len(raw_channel_data), fsBionode)
# eeg_alpha = bandpass_filter_alpha(raw_channel_data, fsBionode)
# time_min, alpha_power = compute_alpha_power(eeg_alpha, fsBionode, 1)
# smoothed_power = smooth_alpha_power(alpha_power, fsBionode, 1)
# time_sec_alpha = time_min * 60

# # === Video frame queue ===
# queue_frame = Queue(maxsize=1)

# # === Launch video processing in separate thread ===
# def run_video():
#     cap = cv2.VideoCapture(video_path)
#     gaze = MediaPipeGazeTracking()
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = 0
#     while cap.isOpened():
#         if paused[0]:
#             cv2.waitKey(1)
#             continue
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (960, 720))
#         current_time = frame_count / fps
#         video_frame_time[0] = current_time  # Sync time anchor
#         frame_count += 1
#         gaze.refresh(frame)
#         gaze.is_blinking(current_time)
#         annotated = gaze.annotated_frame(current_time)
#         if queue_frame.empty():
#             queue_frame.put(annotated)
#         cv2.waitKey(1)
#     cap.release()
#     gaze.export_to_csv()

# video_thread = Thread(target=run_video, daemon=True)
# video_thread.start()

# # === Plotting ===
# fig = plt.figure(figsize=(16, 8))
# gs = fig.add_gridspec(2, 2)
# ax_video = fig.add_subplot(gs[0, 0])
# ax_eeg = fig.add_subplot(gs[0, 1])
# ax_alpha = fig.add_subplot(gs[1, :])

# video_image = ax_video.imshow(np.zeros((720, 960, 3), dtype=np.uint8))
# ax_video.axis('off')
# ax_video.set_title("Video Feed (Gaze Tracked)")

# line, = ax_eeg.plot([], [], color='blue')
# event_dots, = ax_eeg.plot([], [], 'ro')
# ax_eeg.set_xlabel("Time (s)")
# ax_eeg.set_ylabel("Voltage (V)")
# ax_eeg.set_title("Filtered EEG with Eye Movement Detection")
# ax_eeg.grid(True)
# ax_eeg.set_xlim(0, window_sec)
# ax_eeg.set_ylim(-0.00007, 0.00007)

# raw_line, = ax_alpha.plot([], [], label='Raw Alpha Power (V²)', color='green', alpha=0.5)
# smooth_line, = ax_alpha.plot([], [], label='Smoothed Alpha Power', color='red', linewidth=2)
# ax_alpha.set_xlabel('Time (s)')
# ax_alpha.set_ylabel('Alpha Power (V²)')
# ax_alpha.set_yscale('log')
# ax_alpha.set_title('Animated Alpha Power (20s Window)')
# ax_alpha.grid(True)
# ax_alpha.legend()

# text_labels = []

# def detect_eye_movements(y_win, t_win):
#     threshold_spike = 0.0005
#     threshold_dip = -0.0001
#     max_gap_sec = 0.12
#     max_gap_samples = int(max_gap_sec * fsBionode)
#     events = []
#     i = 0
#     while i < len(y_win) - max_gap_samples:
#         if y_win[i] > threshold_spike:
#             for j in range(i + 1, min(i + max_gap_samples, len(y_win))):
#                 if y_win[j] < threshold_dip:
#                     mid_idx = i + (j - i) // 2
#                     events.append((t_win[mid_idx], y_win[mid_idx]))
#                     i = j + 1
#                     break
#             else:
#                 i += 1
#         else:
#             i += 1
#     return events

# def init():
#     line.set_data([], [])
#     event_dots.set_data([], [])
#     video_image.set_array(np.zeros((720, 960, 3), dtype=np.uint8))
#     raw_line.set_data([], [])
#     smooth_line.set_data([], [])
#     ax_alpha.set_xlim(0, 20)
#     ax_alpha.set_ylim(np.min(alpha_power), np.max(alpha_power))
#     return [line, event_dots, video_image, raw_line, smooth_line]

# def update(_):
#     global text_labels
#     if paused[0]:
#         return [line, event_dots, video_image, raw_line, smooth_line] + text_labels

#     current_time = video_frame_time[0]

#     # === EEG update ===
#     eeg_mask = (time >= current_time) & (time <= current_time + window_sec)
#     if np.any(eeg_mask):
#         t_win = time[eeg_mask]
#         y_win = filtered[eeg_mask]
#         t_win_relative = t_win - t_win[0]
#         line.set_data(t_win_relative, y_win)
#         ax_eeg.set_title(f"Filtered EEG ({t_win[0]:.1f}s - {t_win[-1]:.1f}s)")

#         events = detect_eye_movements(y_win, t_win)
#         t_events, y_events = [], []
#         if events:
#             abs_t_events, y_events = zip(*events)
#             t_events = [t - t_win[0] for t in abs_t_events]
#             event_dots.set_data(t_events, y_events)
#         else:
#             event_dots.set_data([], [])
#         for txt in text_labels:
#             if txt in ax_eeg.texts:
#                 txt.remove()
#         text_labels = []
#         for tx, ty in zip(t_events, y_events):
#             txt = ax_eeg.text(tx, ty + 0.00004, 'Eye Movement/EOG', color='red', fontsize=8)
#             text_labels.append(txt)

#     # === Alpha power update ===
#     t_start = max(current_time - 20, 0)
#     t_end = current_time
#     try:
#         alpha_mask = (time_sec_alpha >= current_time) & (time_sec_alpha <= current_time + window_sec)
#         if np.any(alpha_mask):
#             raw_line.set_data(time_sec_alpha[alpha_mask] - current_time, alpha_power[alpha_mask])
#             smooth_line.set_data(time_sec_alpha[alpha_mask] - current_time, smoothed_power[alpha_mask])
#             ax_alpha.set_xlim(0, window_sec)
#     except Exception as e:
#         print(f"Alpha plot error at time {current_time:.2f}s: {e}")

#     # === Video ===
#     if not queue_frame.empty():
#         frame = queue_frame.get()
#         video_image.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     print(f"Video Time: {current_time:.2f}s | EEG Window: {t_win[0]:.2f}s | Alpha Power Window: {t_start:.2f}-{t_end:.2f}s")

#     return [line, event_dots, video_image, raw_line, smooth_line] + text_labels

# def on_key(event):
#     if event.key == ' ':
#         paused[0] = not paused[0]
#         print("Paused" if paused[0] else "Resumed")

# fig.canvas.mpl_connect('key_press_event', on_key)

# ani = animation.FuncAnimation(
#     fig,
#     update,
#     init_func=init,
#     interval=step_sec * 1000,
#     blit=True,
#     cache_frame_data=False
# )

# plt.tight_layout()
# plt.show()


# eeg_gaze_alpha_plot.py

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
# File paths and parameters for EEG and video processing
filename = blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
ADCres = 12  # ADC resolution in bits
fsBionode = 6250  # Sampling frequency in Hz
channel = 1  # EEG channel to analyze
window_sec = 20  # Time window to display (seconds)
step_sec = 0.02  # Animation update interval (seconds)
video_path = "video_recordings/alessandro_edit.mp4"  # Video file path

# === Pause flag ===
# Using lists to allow modification across threads
paused = [False]  # Global pause state
video_frame_time = [0.0]  # Current video time (acts as timeline anchor)

# === Load and preprocess EEG ===
# Load raw EEG data from binary file
data = fn_BionodeBinOpen(filename, ADCres, fsBionode)
# Convert to float32 and scale to volts
rawCha = data["channelsData"].astype(np.float32)
rawCha = (rawCha - 2**11) * 1.8 / (2**12 * 1000)  # Convert ADC values to volts

# Apply low-pass filter to remove high-frequency noise
highCutoff = 60  # Cutoff frequency in Hz
b, a = butter(4, highCutoff / (fsBionode / 2), btype='low')  # 4th order Butterworth
filtered = filtfilt(b, a, rawCha[channel])  # Zero-phase filtering
time = np.arange(len(filtered)) / fsBionode  # Create time array in seconds

# === Alpha Power Computation ===
# Note: This appears to be redundant with the above EEG loading - uses same file but processes differently
raw_channel_data = load_and_preprocess_data(blockPath, ADCres, fsBionode, channel)
duration_sec = print_data_stats(len(raw_channel_data), fsBionode)
eeg_alpha = bandpass_filter_alpha(raw_channel_data, fsBionode)  # Get alpha band (8-13Hz)
time_min, alpha_power = compute_alpha_power(eeg_alpha, fsBionode, 1)  # Compute power
smoothed_power = smooth_alpha_power(alpha_power, fsBionode, 1)  # Smooth the power
time_sec_alpha = time_min * 60  # Convert minutes to seconds

# === Video frame queue ===
# Thread-safe queue for passing video frames between threads
queue_frame = Queue(maxsize=1)  # Only holds latest frame

# === Launch video processing in separate thread ===
def run_video():
    """Thread function that processes video frames and performs gaze tracking"""
    cap = cv2.VideoCapture(video_path)
    gaze = MediaPipeGazeTracking()  # Gaze tracking object
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get video FPS
    frame_count = 0
    
    while cap.isOpened():
        if paused[0]:  # Check pause state
            cv2.waitKey(1)
            continue
            
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (960, 540))  # Resize for display
        current_time = frame_count / fps  # Calculate current video time
        video_frame_time[0] = current_time  # Update global time reference
        frame_count += 1
        
        # Process gaze tracking
        gaze.refresh(frame)
        gaze.is_blinking(current_time)
        annotated = gaze.annotated_frame(current_time)  # Get frame with gaze annotations
        
        if queue_frame.empty():  # Only update if queue is empty
            queue_frame.put(annotated)
            
        cv2.waitKey(1)  # Small delay
        
    cap.release()
    gaze.export_to_csv()  # Save gaze data

# Start video processing thread (daemon=True means it will exit when main exits)
video_thread = Thread(target=run_video, daemon=True)
video_thread.start()

# === Plotting ===
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(3, 2)  # 2 rows, 2 columns

# Video subplot (top-left)
ax_video = fig.add_subplot(gs[0, 0])
ax_video.set_aspect('auto')  # or 'equal' or 'box'
video_image = ax_video.imshow(np.zeros((540, 960, 3), dtype=np.uint8))  # Blank image
ax_video.axis('off')
ax_video.set_title("Video Feed (Gaze Tracked)")

# EEG subplot (top-right)
ax_eeg = fig.add_subplot(gs[0, 1])
line, = ax_eeg.plot([], [], color='blue')  # Main EEG line
event_dots, = ax_eeg.plot([], [], 'ro')  # Eye movement events
ax_eeg.set_xlabel("Time (s)")
ax_eeg.set_ylabel("Voltage (V)")
ax_eeg.set_title("Filtered EEG with Eye Movement Detection")
ax_eeg.grid(True)
ax_eeg.set_xlim(0, window_sec)
ax_eeg.set_ylim(-0.00007, 0.00007)  # Fixed y-axis for better visualization


# Alpha power subplot (bottom full width)
ax_alpha = fig.add_subplot(gs[2, :])
raw_line, = ax_alpha.plot([], [], label='Raw Alpha Power (V²)', color='green', alpha=0.5)
smooth_line, = ax_alpha.plot([], [], label='Smoothed Alpha Power', color='red', linewidth=2)
ax_alpha.set_xlabel('Time (s)')
ax_alpha.set_ylabel('Alpha Power (V²)')
ax_alpha.set_yscale('log')  # Logarithmic scale for power
ax_alpha.set_ylim(2,40)
ax_alpha.set_title('Animated Alpha Power (20s Window)')
ax_alpha.grid(True)
ax_alpha.legend()


# Face zoom subplot (bottom-left corner)
ax_face = fig.add_subplot(gs[1, 0])  # Place in bottom-left quadrant
ax_face.set_aspect('auto')  # or 'equal' or 'box'
face_image = ax_face.imshow(np.zeros((40, 112, 3), dtype=np.uint8))  # Placeholder image
ax_face.axis('off')
ax_face.set_title("Zoomed-In Face")


text_labels = []  # Stores text annotations for eye movement events

def detect_eye_movements(y_win, t_win):
    """Detects eye movements in EEG signal using threshold-based approach
    
    Args:
        y_win: EEG signal window (voltage values)
        t_win: Corresponding time values
        
    Returns:
        List of (time, voltage) tuples for detected events
    """
    threshold_spike = 0.0005  # Upper threshold for eye movement
    threshold_dip = -0.0001   # Lower threshold
    max_gap_sec = 0.12        # Maximum allowed time between spike and dip
    max_gap_samples = int(max_gap_sec * fsBionode)
    events = []
    i = 0
    
    # Scan through signal looking for spike-dip patterns
    while i < len(y_win) - max_gap_samples:
        if y_win[i] > threshold_spike:
            # Look for corresponding dip within max_gap_samples
            for j in range(i + 1, min(i + max_gap_samples, len(y_win))):
                if y_win[j] < threshold_dip:
                    # Found a valid event - mark midpoint
                    mid_idx = i + (j - i) // 2
                    events.append((t_win[mid_idx], y_win[mid_idx]))
                    i = j + 1  # Skip ahead to avoid overlapping detections
                    break
            else:
                i += 1
        else:
            i += 1
    return events

def init():
    """Initialize animation with empty plots"""
    line.set_data([], [])
    event_dots.set_data([], [])
    video_image.set_array(np.zeros((540, 960, 3), dtype=np.uint8))
    raw_line.set_data([], [])
    smooth_line.set_data([], [])
    ax_alpha.set_xlim(0, window_sec)
    ax_alpha.set_ylim(2,40)
    return [line, event_dots, video_image, raw_line, smooth_line, face_image]


def update(_):
    """Update function called for each animation frame"""
    global text_labels
    if paused[0]:
        return [line, event_dots, video_image, raw_line, smooth_line] + text_labels

    current_time = video_frame_time[0]  # Get current video time (shared timeline)

    # === EEG update ===
    eeg_mask = (time >= current_time) & (time <= current_time + window_sec)
    if np.any(eeg_mask):
        t_win = time[eeg_mask]
        y_win = filtered[eeg_mask]
        t_win_relative = t_win - t_win[0]  # Make time relative to window start
        line.set_data(t_win_relative, y_win)
        ax_eeg.set_title(f"Filtered EEG ({t_win[0]:.1f}s - {t_win[-1]:.1f}s)")

        # Detect eye movements and plot them
        events = detect_eye_movements(y_win, t_win)
        t_events, y_events = [], []
        if events:
            abs_t_events, y_events = zip(*events)
            t_events = [t - t_win[0] for t in abs_t_events]  # Make times relative
            event_dots.set_data(t_events, y_events)
        else:
            event_dots.set_data([], [])
            
        # Clear previous text labels
        for txt in text_labels:
            if txt in ax_eeg.texts:
                txt.remove()
        text_labels = []
        
        # Add new labels for detected events
        for tx, ty in zip(t_events, y_events):
            txt = ax_eeg.text(tx, ty + 0.00004, 'Eye Movement/EOG', color='red', fontsize=8)
            text_labels.append(txt)

    # === Alpha power update ===
    t_start = max(current_time - window_sec, 0)
    t_end = current_time
    try:
        alpha_mask = (time_sec_alpha >= current_time) & (time_sec_alpha <= current_time + window_sec)
        if np.any(alpha_mask):
            raw_line.set_data(time_sec_alpha[alpha_mask] - current_time, alpha_power[alpha_mask])
            smooth_line.set_data(time_sec_alpha[alpha_mask] - current_time, smoothed_power[alpha_mask])
            ax_alpha.set_xlim(0, window_sec)
            ax_alpha.set_ylim(2,40)
    except Exception as e:
        print(f"Alpha plot error at time {current_time:.2f}s: {e}")

    # === Video ===
    if not queue_frame.empty():
        frame = queue_frame.get()
        video_image.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
                # Extract and display zoomed-in face region
        x1, y1, x2, y2 = 528, 230, 640, 270 #640-528=112, 270-230=40
        face_crop = frame[y1:y2, x1:x2]
        face_image.set_array(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

        #528, 324
        #640, 324
        #640, 350
        #528, 350
    

    print(f"Video Time: {current_time:.2f}s | EEG Window: {t_win[0]:.2f}s | Alpha Power Window: {t_start:.2f}-{t_end:.2f}s")

    return [line, event_dots, video_image, raw_line, smooth_line, face_image] + text_labels

def on_key(event):
    """Keyboard callback for pause/resume"""
    if event.key == ' ':
        paused[0] = not paused[0]
        print("Paused" if paused[0] else "Resumed")

fig.canvas.mpl_connect('key_press_event', on_key)

# Create animation
ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=step_sec * 1000,  # Convert to milliseconds
    blit=True,  # Optimize drawing
    cache_frame_data=False  # Don't cache frames (important for real-time)
)

plt.tight_layout()
plt.show()