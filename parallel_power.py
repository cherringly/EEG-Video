
import numpy as np
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

# EEG CONFIG
channel = 1
fsBionode = 25e3 / 2  # 12.5 kHz
ADCres = 12
window_sec = 10
window_size = 0.2
step_sec = 0.02
blockPath = '/Users/maryzhang/Downloads/EarEEG_Matlab/Data/25-31-03/2025.03.31.17.53.13_AAO-18.bin'

# VIDEO CONFIG
video_path = "video_recordings/alessandro.mov"

# EEG PROCESSING
def preprocess_eeg():
    rawCha = np.array(fn_BionodeBinOpen(blockPath, ADCres, fsBionode)['channelsData'])
    rawCha = (rawCha - 2**11) * 1.8 / (2**12)
    rawCha = np.nan_to_num(rawCha, nan=0.0, posinf=0.0, neginf=0.0)

    # sos_lp = signal.butter(4, 50, btype='low', fs=fsBionode, output='sos')
    # eeg_signal = signal.sosfiltfilt(sos_lp, rawCha[channel - 1])

    sos_alpha = signal.butter(4, [8, 12], btype='bandpass', fs=fsBionode, output='sos')
    eeg_alpha = signal.sosfiltfilt(sos_alpha, rawCha[channel - 1])
    return eeg_alpha

# Shared data
queue_frame = Queue()
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

        frame = cv2.resize(frame, (960, 720))
        current_time = frame_count / fps
        frame_count += 1

        gaze.refresh(frame)
        gaze.is_blinking(current_time)
        annotated = gaze.annotated_frame(current_time)
        queue_frame.put(annotated)

    cap.release()
    gaze.export_to_csv()

# EEG PLOT
window_samples = int(window_size * fsBionode)
step_samples = int(step_sec * fsBionode)

# fig, (ax_video, ax_eeg) = plt.subplots(2, 1, figsize=(12, 8))
# line, = ax_eeg.plot([], [], animated=True)
# ax_eeg.set_xscale('linear')
# ax_eeg.set_yscale('log')
# ax_eeg.set_xlabel('Frequency (Hz)')
# ax_eeg.set_ylabel('PSD (log scale, uV²/Hz)')
# ax_eeg.set_title('EEG Alpha Band PSD (Welch Method)')

# img_disp = ax_video.imshow(np.zeros((720, 960, 3), dtype=np.uint8))
# ax_video.axis('off')

# index = [0]
# def update(frame):
#     # EEG PSD update
#     idx = index[0]
#     if idx + window_samples > len(eeg_alpha):
#         return line,
#     segment = eeg_alpha[idx:idx + window_samples]
#     segment = segment * np.hanning(window_samples)
#     freqs, psd = signal.welch(
#         segment,
#         fs=fsBionode,
#         window='hann',
#         nperseg=window_samples,
#         noverlap=window_samples//2,
#         scaling='density'
#     )
#     alpha_band_mask = (freqs >= 8) & (freqs <= 12)
#     line.set_data(freqs[alpha_band_mask], psd[alpha_band_mask])
#     ax_eeg.set_xlim(0, 70)
#     ax_eeg.set_ylim(np.max([np.min(psd[alpha_band_mask]), 1e-12]), np.max(psd[alpha_band_mask]))
#     index[0] += step_samples

#     # VIDEO update
#     if not queue_frame.empty():
#         frame = queue_frame.get()
#         img_disp.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     return line, img_disp

# # Start video processing in a separate thread
# Thread(target=run_video, daemon=True).start()

# ani = animation.FuncAnimation(fig, update, init_func=lambda: (line, img_disp), interval=step_sec * 1000, blit=True)
# plt.tight_layout()
# plt.show()

# Alpha power time buffer for 10s history
power_buffer = deque(maxlen=int(window_size / step_sec))
time_buffer = deque(maxlen=int(window_size / step_sec))

# Modified figure layout to fit 3 subplots
fig, (ax_video, ax_eeg, ax_power) = plt.subplots(3, 1, figsize=(12, 10))

# EEG PSD Plot
line, = ax_eeg.plot([], [], animated=True)
ax_eeg.set_xscale('linear')
ax_eeg.set_yscale('log')
ax_eeg.set_xlabel('Frequency (Hz)')
ax_eeg.set_ylabel('PSD (log scale, uV²/Hz)')
ax_eeg.set_title('EEG Alpha Band PSD (Welch Method)')

# Alpha power over time plot
line_power, = ax_power.plot([], [], color='green', animated=True)
ax_power.set_xlabel('Time (s)')
ax_power.set_ylabel('Alpha Power (V²)')
ax_power.set_title('Alpha Power Over Time (8–12 Hz)')

# Video display
img_disp = ax_video.imshow(np.zeros((720, 960, 3), dtype=np.uint8))
ax_video.axis('off')

# Update function
index = [0]
def update(frame):
    idx = index[0]
    if idx + window_samples > len(eeg_alpha):
        return line, line_power, img_disp

    segment = eeg_alpha[idx:idx + window_samples]
    segment = segment * np.hanning(window_samples)
    freqs, psd = signal.welch(
        segment,
        fs=fsBionode,
        window='hann',
        nperseg=window_samples,
        noverlap=window_samples//2,
        scaling='density'
    )


    # Update PSD line plot
    alpha_mask = (freqs >= 8) & (freqs <= 12)
    print(psd[alpha_mask]+" - psd with alpha mask")
    print(psd + " - psd")
    line.set_data(freqs[alpha_mask], psd[alpha_mask])
    ax_eeg.set_xlim(8, 12)
    ax_eeg.set_ylim(np.max([np.min(psd[alpha_mask])]), np.max(psd[alpha_mask]))

    # Compute integral (alpha power) and append
    alpha_power = simps(psd[alpha_mask], freqs[alpha_mask])  # in V^2
    current_time = idx / fsBionode
    power_buffer.append(alpha_power)
    time_buffer.append(current_time)

    # Update power-over-time plot
    line_power.set_data(list(time_buffer), list(power_buffer))
    ax_power.set_xlim(current_time - window_sec, current_time)
    ax_power.set_ylim(1e-10, max(1e-9, max(power_buffer)))  # log-like scaling

    # Update video frame
    if not queue_frame.empty():
        frame = queue_frame.get()
        img_disp.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    index[0] += step_samples
    return line, line_power, img_disp

# Start video thread
Thread(target=run_video, daemon=True).start()

ani = animation.FuncAnimation(fig, update, init_func=lambda: (line, line_power, img_disp), interval=step_sec * 1000, blit=True)
plt.tight_layout()
plt.show()
