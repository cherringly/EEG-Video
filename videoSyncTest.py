import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
from bionodebinopen import fn_BionodeBinOpen
import cv2
from threading import Thread
from queue import Queue
from gaze_track import MediaPipeGazeTracking

# === CONFIG ===
filename = blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"  # File path
ADCres = 12
fsBionode = 6250
channel = 1
window_sec = 5
step_sec = 0.02
video_path = "video_recordings/alessandro.mov"

# === Pause flag ===
paused = [False]

# === Load and preprocess EEG ===
data = fn_BionodeBinOpen(filename, ADCres, fsBionode)
rawCha = data["channelsData"].astype(np.float32)
rawCha = (rawCha - 2**11) * 1.8 / (2**12 * 1000)
highCutoff = 60
b, a = butter(4, highCutoff / (fsBionode / 2), btype='low')
filtered = filtfilt(b, a, rawCha[channel])
time = np.arange(len(filtered)) / fsBionode

# === Sliding window setup ===
window_samples = int(window_sec * fsBionode)
step_samples = int(step_sec * fsBionode)

# === Video frame queue ===
queue_frame = Queue(maxsize=1)

# === Launch video processing in separate thread ===
def run_video():
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
        frame = cv2.resize(frame, (960, 720))
        current_time = frame_count / fps
        frame_count += 1
        gaze.refresh(frame)
        gaze.is_blinking(current_time)
        annotated = gaze.annotated_frame(current_time)
        if queue_frame.empty():
            queue_frame.put(annotated)
        cv2.waitKey(1)
    cap.release()
    gaze.export_to_csv()

video_thread = Thread(target=run_video, daemon=True)
video_thread.start()

# === Plotting ===
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 2)
ax_video = fig.add_subplot(gs[0, 0])
ax_eeg = fig.add_subplot(gs[0, 1])
video_image = ax_video.imshow(np.zeros((720, 960, 3), dtype=np.uint8))
ax_video.axis('off')
ax_video.set_title("Video Feed (Gaze Tracked)")

line, = ax_eeg.plot([], [], color='blue')
event_dots, = ax_eeg.plot([], [], 'ro')
ax_eeg.set_xlabel("Time (s)")
ax_eeg.set_ylabel("Voltage (V)")
ax_eeg.set_title("Filtered EEG with Eye Movement Detection")
ax_eeg.grid(True)
ax_eeg.set_xlim(0, window_sec)
ax_eeg.set_ylim(-0.00007, 0.00007)  # ZOOMED Y-AXIS HERE

text_labels = []
index = [0]

def detect_movements(y_win, t_win):
    threshold_spike = 0.0005
    threshold_dip = -0.0001
    max_gap_sec = 0.12
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
    line.set_data([], [])
    event_dots.set_data([], [])
    global text_labels
    for txt in text_labels:
        if txt in ax_eeg.texts:
            txt.remove()
    text_labels = []
    video_image.set_array(np.zeros((720, 960, 3), dtype=np.uint8))
    ax_eeg.set_xlim(0, window_sec)
    ax_eeg.set_ylim(-0.00007, 0.00007)  # ZOOMED Y-AXIS HERE
    return line, event_dots, video_image

def update(frame):
    global text_labels

    if paused[0]:
        return [line, event_dots, video_image] + text_labels

    idx = index[0]
    if idx + window_samples <= len(filtered):
        t_win = time[idx:idx + window_samples]
        y_win = filtered[idx:idx + window_samples]
        start_time = t_win[0]
        t_win_relative = t_win - start_time
        line.set_data(t_win_relative, y_win)
        ax_eeg.set_xlim(0, window_sec)
        ax_eeg.set_ylim(-0.00007, 0.00007)  # ZOOMED Y-AXIS HERE
        ax_eeg.set_title(f"Filtered EEG ({start_time:.1f}s - {start_time + window_sec:.1f}s)")
        events = detect_eye_movements(y_win, t_win)
        if events:
            abs_t_events, y_events = zip(*events)
            t_events = [t - start_time for t in abs_t_events]
            event_dots.set_data(t_events, y_events)
        else:
            event_dots.set_data([], [])
            t_events, y_events = [], []
        for txt in text_labels:
            if txt in ax_eeg.texts:
                txt.remove()
        text_labels = []
        for tx, ty in zip(t_events, y_events):
            txt = ax_eeg.text(tx, ty + 0.00004, 'Eye Movement/EOG', color='red', fontsize=8)  # ADJUSTED TEXT OFFSET
            text_labels.append(txt)
        index[0] += step_samples

    if not queue_frame.empty():
        frame = queue_frame.get()
        video_image.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return [line, event_dots, video_image] + text_labels

def force_initial_limits():
    ax_eeg.set_xlim(0, window_sec)
    ax_eeg.set_ylim(-0.00007, 0.00007)  # ZOOMED Y-AXIS HERE
    fig.canvas.draw()
    plt.pause(0.1)

def on_key(event):
    if event.key == ' ':
        paused[0] = not paused[0]
        print("Paused" if paused[0] else "Resumed")

fig.canvas.mpl_connect('key_press_event', on_key)

force_initial_limits()

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