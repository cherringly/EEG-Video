'''
TODO: NOT COMPLETED
- Having trouble with the sliding view


Real-time EEG and Gaze Tracking Visualization
1. Real Time EEG Simulation
- Loads EEG data from .bin file
- Sliding buffer for latest EEG data

2. Real Time Gaze Tracking
- Uses MediaPipe for eye landmarks
- Calculates Eye Aspect Ratio (EAR) to detect blinks

3. Real Time STFT
- Computes STFT of EEG data

4. Live Spectrogram
- Top: video feed with gaze tracking
- Bottom: real-time EEG spectrogram of alpha band

5. Export
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy import signal
from collections import deque
from gaze_track import MediaPipeGazeTracking
from bionodebinopen import fn_BionodeBinOpen

# Load EEG
fsBionode = 12_500  # Half of 25kHz
ADCres = 12
channel = 1
blockPath = '/Users/maryzhang/Downloads/EarEEG_Matlab/Data/25-31-03/2025.03.31.17.53.13_AAO-18.bin'
earData = fn_BionodeBinOpen(blockPath, ADCres, fsBionode)
rawChaB = (np.array(earData['channelsData']) - 2**11) * 1.8 / (2**12 * 1000)

# Preprocess
bB, aB = signal.butter(4, 20 / (fsBionode / 2), btype='low')
PPfiltChaB = signal.filtfilt(bB, aB, rawChaB[channel-1, :])

# Video init
cap = cv2.VideoCapture("video_recordings/alessandro.mov")
fps = cap.get(cv2.CAP_PROP_FPS)
fcount = 0
gaze = MediaPipeGazeTracking()

# STFT params
win_sec = 0.5
step_sec = 0.05
win_samples = int(win_sec * fsBionode)
step_samples = int(step_sec * fsBionode)
nperseg = win_samples
noverlap = win_samples - step_samples

eeg_buffer = deque(maxlen=10 * fsBionode)

# Create OpenCV display window
cv2.namedWindow("Gaze + EEG Spectrogram", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gaze + EEG Spectrogram", 1920, 1080)

fig, ax2 = plt.subplots(figsize=(10, 4))
canvas = FigureCanvas(fig)

spectrogram_img = None

while True:
    start_time = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret:
        break

    current_time = fcount / fps
    fcount += 1

    gaze.refresh(frame)
    annotated = gaze.annotated_frame(current_time)

    # Ensure blinking logic executes correctly
    _ = gaze.is_blinking(current_time)

    # Simulate real-time EEG (synchronized with video)
    eeg_index = int(current_time * fsBionode)
    next_index = eeg_index + step_samples
    if next_index < len(PPfiltChaB):
        eeg_buffer.extend(PPfiltChaB[eeg_index:next_index])

    if len(eeg_buffer) >= win_samples:
        eeg_array = np.array(eeg_buffer)
        f, t, Zxx = signal.stft(eeg_array, fs=fsBionode, nperseg=nperseg, noverlap=noverlap)
        freq_mask = (f >= 8) & (f <= 12)
        f = f[freq_mask]
        power = np.abs(Zxx[freq_mask])**2

        ax2.clear()
        ax2.set_ylim(8, 12)
        ax2.set_ylabel('Frequency [Hz]')
        ax2.set_xlabel('Time [s]')
        ax2.set_title('Real-Time EEG Spectrogram (Alpha Band 8-12Hz)')
        ax2.pcolormesh(t - t[-1], f, 10 * np.log10(power + 1e-10), shading='gouraud', cmap='jet')

        canvas.draw()
        s, (width, height) = canvas.print_to_buffer()
        eeg_img = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        eeg_bgr = cv2.cvtColor(eeg_img, cv2.COLOR_RGBA2BGR)
        eeg_bgr = cv2.resize(eeg_bgr, (annotated.shape[1], annotated.shape[0]))

        combined = np.vstack((annotated, eeg_bgr))
        cv2.imshow("Gaze + EEG Spectrogram", combined)
    else:
        cv2.imshow("Gaze + EEG Spectrogram", annotated)

    key = cv2.waitKey(max(1, int(1000 / fps)))
    if key & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
gaze.export_to_csv()
