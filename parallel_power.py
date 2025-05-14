# parallel_power.py

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bionodebinopen import fn_BionodeBinOpen
import cv2
import mediapipe as mp
import time
from threading import Thread
from queue import Queue

from gaze_track import MediaPipeGazeTracking

# EEG CONFIG
channel = 1
fsBionode = 25e3 / 2  # 12.5 kHz
ADCres = 12
window_sec = 10
step_sec = 0.05
blockPath = '/Users/maryzhang/Downloads/EarEEG_Matlab/Data/25-31-03/2025.03.31.17.53.13_AAO-18.bin'

# VIDEO CONFIG
video_path = "video_recordings/alessandro.mov"

# EEG PROCESSING
def preprocess_eeg():
    rawCha = np.array(fn_BionodeBinOpen(blockPath, ADCres, fsBionode)['channelsData'])
    rawCha = (rawCha - 2**11) * 1.8 / (2**12 * 1000)
    rawCha = np.nan_to_num(rawCha, nan=0.0, posinf=0.0, neginf=0.0)

    sos_lp = signal.butter(4, 50, btype='low', fs=fsBionode, output='sos')
    eeg_signal = signal.sosfiltfilt(sos_lp, rawCha[channel - 1])

    sos_alpha = signal.butter(4, [8, 12], btype='bandpass', fs=fsBionode, output='sos')
    eeg_alpha = signal.sosfiltfilt(sos_alpha, eeg_signal)
    return eeg_alpha

# Shared data
queue_frame = Queue()
queue_eeg = Queue()
eeg_alpha = preprocess_eeg()

# VIDEO THREAD
def run_video():
    cap = cv2.VideoCapture(video_path)
    gaze = MediaPipeGazeTracking()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 720))  # Make video larger and wider
        current_time = frame_count / fps
        frame_count += 1

        gaze.refresh(frame)
        gaze.is_blinking(current_time)
        annotated = gaze.annotated_frame(current_time)
        queue_frame.put(annotated)

    cap.release()
    gaze.export_to_csv()

# EEG PLOT
window_samples = int(window_sec * fsBionode)
step_samples = int(step_sec * fsBionode)

fig, (ax_video, ax_eeg) = plt.subplots(2, 1, figsize=(12, 8))
line, = ax_eeg.plot([], [], animated=True)
ax_eeg.set_ylim(np.nanmin(eeg_alpha), np.nanmax(eeg_alpha))
ax_eeg.set_xlabel('Time (s)')
ax_eeg.set_ylabel('Voltage (V)')
ax_eeg.set_title('EEG Alpha Band (8-12 Hz)')

img_disp = ax_video.imshow(np.zeros((720, 960, 3), dtype=np.uint8))
ax_video.axis('off')

index = [0]
def update(frame):
    # EEG update
    idx = index[0]
    if idx + window_samples > len(eeg_alpha):
        return line,
    segment = eeg_alpha[idx:idx + window_samples]
    start_time = idx / fsBionode
    time_window = np.arange(window_samples) / fsBionode + start_time
    line.set_data(time_window, segment)
    ax_eeg.set_xlim(time_window[0], time_window[-1])
    index[0] += step_samples

    # VIDEO update
    if not queue_frame.empty():
        frame = queue_frame.get()
        img_disp.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return line, img_disp

# Start video processing in a separate thread
Thread(target=run_video, daemon=True).start()

ani = animation.FuncAnimation(fig, update, init_func=lambda: (line, img_disp), interval=step_sec * 1000, blit=True)
plt.tight_layout()
plt.show()