import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt, sosfiltfilt
from threading import Thread
from queue import Queue, Full
from bionodebinopen import fn_BionodeBinOpen
from movement_track import HeadJawTracker
from tdt import read_block

# === CONFIG ===
# BLOCK_PATH_NEUROPULSE = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
BLOCK_PATH_TDT = r"\Users\maryz\EEG-Video\SubjectG-250331-160838"
VIDEO_PATH = r"video_recordings/4.08_tdt_e.mp4"
ADC_RES = 12
TDT_FS = 12207
# FS = 5537
CHANNEL = 1
WINDOW_SEC = 10
EMG_YLIM = (-0.0002, 0.0002) #only for tdt
JAW_BOX_SIZE = 120  # size of zoomed jaw crop
VIDEO_OFFSET = 1
# 2.2 for 3.52
# 4.5 for 4.08

paused = [False]
pause_start = None
t0_real = None
queue_frame = Queue(maxsize=2)
jaw_windows = []
drawn_jaw_idx = 0

# === Load & Filter EMG ===
# neuropulse data processing
# data = fn_BionodeBinOpen(BLOCK_PATH_NEUROPULSE, ADC_RES, FS)
# raw = np.array(data['channelsData'])
# scale = 1.8 / 4096.0 / 10000  # V per unit / gain
# emg = (raw - 2048) * scale
# rawCha = np.nan_to_num(emg[CHANNEL])


# TDT data processing
data = read_block(BLOCK_PATH_TDT)
raw = data.streams.EEGw.data
rawC = raw[1:].astype(np.float32)
# rawC = (rawC - 2048) * 1.8 / (2**ADCres * 1000)  # Scale to volts
rawC = np.nan_to_num(rawC)
rawCha = rawC[CHANNEL]

b, a = butter(4, 150 / (TDT_FS / 2), btype='low')
emg_filtered = filtfilt(b, a, rawCha)


# b,a = butter(4, [0.1, 50], btype='bandpass', fs=TDT_FS)
# emg_filtered = filtfilt(b, a, rawCha)

# def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq

#     sos = butter(order, [low, high], btype='band', output='sos')
#     y = sosfiltfilt(sos, data)
#     return y
# emg_filtered = butter_bandpass_filter(rawCha, 0.1, 50, FS)
time_arr = np.arange(len(emg_filtered)) / TDT_FS
print(f"EMG data loaded: {len(emg_filtered)} samples, {len(emg_filtered) / TDT_FS:.2f} seconds")
print(f"EMG filtered range: min={np.min(emg_filtered):.2e}, max={np.max(emg_filtered):.2e}")
print(f"EMG filtered random: {emg_filtered[550:560]}")
print(np.isnan(raw).any(), np.isnan(rawCha).any())
print(np.nanmin(raw), np.nanmax(raw))


# === Video Thread ===
def run_video():
    global jaw_windows, t0_real
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_MSEC, VIDEO_OFFSET * 1000)
    tracker = HeadJawTracker()
    moving_jaw = False
    jaw_start = 0

    while cap.isOpened():
        if paused[0]:
            cv2.waitKey(1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        msec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        video_time = msec + VIDEO_OFFSET

        if t0_real is None:
            t0_real = time.time() - video_time
        delay = t0_real + video_time - time.time()
        if delay > 0:
            time.sleep(delay)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            frame, pitch, yaw, roll, jaw_state = tracker.process(frame, lm)

            if not moving_jaw and jaw_state != "Neutral":
                jaw_start = video_time
                moving_jaw = True
            elif moving_jaw and jaw_state == "Neutral":
                jaw_windows.append((jaw_start, video_time))
                moving_jaw = False

            h, w, _ = frame.shape
            cx = int(lm.landmark[0].x * w)
            cy = int(lm.landmark[0].y * h)
            x1, x2 = max(cx - JAW_BOX_SIZE, 0), min(cx + JAW_BOX_SIZE, w)
            y1, y2 = max(cy - JAW_BOX_SIZE, 0), min(cy + JAW_BOX_SIZE, h)
            jaw_crop = frame[y1:y2, x1:x2]
            # for idx in [0, 14, 17, 61, 310]:
            for idx in [127, 93, 113, 58, 172, 136, 150, 149, 176, 148, 152,
                        377, 400, 378, 379, 365, 397, 367, 435, 366, 447, 356,
                        134, 131, 203, 206, 216, 212,
                        363, 360, 423, 426, 436, 432,
                        40, 39, 37, 0, 267, 269, 270, 409, 291, 292,
                        61, 146, 91, 181, 84, 17, 314, 405, 321, 308, 324,
                        13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 61, 146, 91, 181, 84,
                        33, 160, 158, 133, 153, 144, 145, 163, 7, 246,
                        263, 362, 385, 387, 373, 380, 381, 382, 466, 388]:
                px = int(lm.landmark[idx].x * w)
                py = int(lm.landmark[idx].y * h)
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
                if y1 <= py < y2 and x1 <= px < x2:
                    cv2.circle(jaw_crop, (px - x1, py - y1), 2, (0, 255, 0), -1)
            jaw_zoom = cv2.resize(jaw_crop, (240, 240))
        else:
            jaw_zoom = np.zeros((240, 240, 3), dtype=np.uint8)

        try:
            queue_frame.put_nowait((frame, jaw_zoom, video_time))
        except Full:
            pass

    cap.release()

Thread(target=run_video, daemon=True).start()

# === Plot Setup ===
fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])

ax_video = fig.add_subplot(gs[0, 0])
ax_video.axis('off')
img_disp = ax_video.imshow(np.zeros((480, 853, 3), dtype=np.uint8))
ax_video.set_title("Video Frame")

ax_zoom = fig.add_subplot(gs[1, 0])
ax_zoom.axis('off')
img_zoom = ax_zoom.imshow(np.zeros((240, 240, 3), dtype=np.uint8))
ax_zoom.set_title("Jaw Zoom")

ax_emg = fig.add_subplot(gs[:, 1])
line_emg, = ax_emg.plot([], [], lw=2, label="EMG")
ax_emg.set_xlabel("Time (s)")
ax_emg.set_ylabel("Voltage (V)")
# ax_emg.set_ylim(-0.01, 0.01)  # Adjusted for better visibility
ax_emg.set_ylim(EMG_YLIM)
# ax_emg.grid(True)

highlighted = []

def init():
    line_emg.set_data([], [])
    return img_disp, img_zoom, line_emg

def update(_):
    global drawn_jaw_idx
    if paused[0] or queue_frame.empty():
        return img_disp, img_zoom, line_emg

    frame, zoom, vtime = queue_frame.get()
    img_disp.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_zoom.set_array(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))

    t1 = vtime
    t0 = max(0, t1 - WINDOW_SEC)
    i0, i1 = int(t0 * TDT_FS), int(t1 * TDT_FS)
    if i1 > len(emg_filtered): return img_disp, img_zoom, line_emg

    t_win = time_arr[i0:i1]
    y_win = emg_filtered[i0:i1]
    line_emg.set_data(t_win, y_win)
    ax_emg.set_xlim(t0, t1)

    for start, end in jaw_windows[drawn_jaw_idx:]:
        if start > t1: break
        if end < t0: continue
        shaded = ax_emg.axvspan(max(start, t0), min(end, t1), color='orange', alpha=0.3, label='Jaw Active')
        highlighted.append(shaded)
    drawn_jaw_idx = len(jaw_windows)

    return img_disp, img_zoom, line_emg

def on_key(event):
    global pause_start, t0_real
    if event.key == ' ':
        paused[0] = not paused[0]
        if paused[0]:
            pause_start = time.time()
        else:
            t0_real += time.time() - pause_start

def plot_static():
    fig, ax = plt.subplots(figsize=(12, 4))
    
    start_sec = 120
    end_sec = 155
    i0 = int(start_sec * TDT_FS)
    i1 = int(end_sec * TDT_FS)
    
    t = time_arr[i0:i1]
    y = emg_filtered[i0:i1]
    
    ax.plot(t, y, lw=3, color="black", label='EMG')
    ax.set_xlim(start_sec, end_sec)
    ax.set_ylim(EMG_YLIM)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'EMG from {start_sec}s to {end_sec}s')
    
    for start, end in jaw_windows:
        if end < start_sec: continue
        if start > end_sec: break
        ax.axvspan(max(start, start_sec), min(end, end_sec), color='blue', alpha=0.3)

    plt.tight_layout()
    plt.show()


fig.canvas.mpl_connect('key_press_event', on_key)
ani = animation.FuncAnimation(fig, update, init_func=init, interval=20, blit=False)

plt.tight_layout()
plt.show()

plot_static()


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02}:{s:02}"

def print_emg_summary():
    total_duration = 0
    print("\nJaw Movement / EMG Periods:")
    for start, end in jaw_windows:
        duration = end - start
        total_duration += duration
        print(f"[{format_time(start)} - {format_time(end)}] {int(duration)} seconds - EMG")
    print(f"TOTAL: {int(total_duration)} seconds - EMG")

print_emg_summary()

