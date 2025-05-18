import numpy as np
import scipy.signal as signal
from scipy.integrate import simpson as simps
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bionodebinopen import fn_BionodeBinOpen
import cv2
from threading import Thread
from queue import Queue
from collections import deque
from gaze_track import MediaPipeGazeTracking

# EEG CONFIG
channel = 0 #channel number (change to 1 for matlab)
fsBionode = 25e3 / 2  #sampling rate
ADCres = 12 #if 12 bits, 2^12 = 4096
window_sec = 10 #number of seconds for the viewing window
window_size = 1 #window size for PSD calculation 
# 1/window_size = # Hz resolution for alpha band PSD
# 1/0.2 = 5 Hz resolution; [0,5,10,15,...], but only 8-12 Hz is used, so only [10] Hz bin
# 1/1 = 1 Hz resolution; [0,1,2,3,4,5,6,7,8,9,10,11,12,13,...], but only 8-12 Hz is used, so only [8,9,10,11,12] Hz bins
step_sec = 0.02 #step size for PSD calculation
blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin" #path to the bin file

# VIDEO CONFIG
video_path = "video_recordings/alessandro.mov" #path to the video file

# EEG PROCESSING
def preprocess_eeg():
    #rawCha is a 2D array with shape (num_channels, total_samples)
    rawCha = np.array(fn_BionodeBinOpen(blockPath, ADCres, fsBionode)['channelsData'])
    
    #ADC outputs unsigned integers (0–4095 for ADCres=12)
    #Subtracting 2048 (2¹¹) centers the signal around zero (range: -2048 to +2047)
    #1.8 V is reference voltage of the ADC
    #Divide by 2^12 to convert to volts (0–1.8 V)
    rawCha = (rawCha - 2**11) * 1.8 / (2**12)
    
    #replaces NaN and inf values with 0
    rawCha = np.nan_to_num(rawCha, nan=0.0, posinf=0.0, neginf=0.0)

    #4th order low-pass Butterworth filter (higher = more attenuation of stopband)
    #sampling rate is passed into function because normalized frequency = (cutoff_frequency) / (fs/2)
    #sos = 2nd order sections because 4th order filter is unstable (precision errors)
    sos_alpha = signal.butter(4, [8, 12], btype='bandpass', fs=fsBionode, output='sos')
    #sosfiltfilt is zero-phase filtering (no phase distortion so preserves timing)
    eeg_alpha = signal.sosfiltfilt(sos_alpha, rawCha[channel])
    
    print("Loaded EEG shape:", rawCha.shape)
    print("Raw EEG preview:", rawCha[channel][:10])

    return rawCha[channel], eeg_alpha




# Shared data
#ensures only the latest video frame is kept (drops older frames if the main thread is busy)
queue_frame = Queue(maxsize=1)
raw_eeg, eeg_alpha = preprocess_eeg()

# VIDEO THREAD
def run_video():
    cap = cv2.VideoCapture(video_path)
    gaze = MediaPipeGazeTracking()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        print(f"[Video] Frame {frame_count}, Time: {current_time:.2f}s, Read successful: {ret}")
        if frame is None or frame.sum() == 0:
            print(f"[Video] Warning: Empty or black frame at count {frame_count}")

        if not ret:
            break

        frame = cv2.resize(frame, (960, 720))
        current_time = frame_count / fps
        frame_count += 1

        gaze.refresh(frame)
        gaze.is_blinking(current_time)
        annotated = gaze.annotated_frame(current_time)
        
        # Only put new frame if queue is empty to prevent blocking
        if queue_frame.empty():
            print("[Video] Inserting frame into queue.")


    cap.release()
    gaze.export_to_csv()

# EEG PLOT CONFIG
window_samples = int(window_size * fsBionode) #0.2*12500 = 2,500 samples (num of data points for PSD)
step_samples = int(step_sec * fsBionode) #0.02*12500 = 250 samples (step size for PSD) (50 fps)
power_buffer = deque(maxlen=int(window_sec / step_sec)) #stores last 10 seconds of power values
time_buffer = deque(maxlen=int(window_sec / step_sec)) #stores last 10 seconds of time values

# Create figure with proper layout
fig = plt.figure(figsize=(15, 10), constrained_layout=True) 
gs = fig.add_gridspec(2, 2)
ax_video = fig.add_subplot(gs[0, 0]) #video feed top left
ax_raw = fig.add_subplot(gs[0, 1]) #raw EEG top right (V over time)
ax_eeg = fig.add_subplot(gs[1, 0]) #PSD plot bottom left (V²/Hz over frequency)
ax_power = fig.add_subplot(gs[1, 1]) #alpha power plot bottom right (V² over time)

# Video display setup
video_image = ax_video.imshow(np.zeros((720, 960, 3), dtype=np.uint8))
ax_video.axis('off')
ax_video.set_title('Video Feed')

# Raw EEG plot setup
raw_line, = ax_raw.plot([], [], 'b-', linewidth=0.5)
ax_raw.set_xlabel('Time (s)')
ax_raw.set_ylabel('Amplitude (V)')
ax_raw.set_title('Raw EEG Signal')
ax_raw.grid(True)

# PSD plot setup
psd_line, = ax_eeg.plot([], [], 'r-')
ax_eeg.set_xscale('linear')
ax_eeg.set_yscale('log')
ax_eeg.set_xlabel('Frequency (Hz)')
ax_eeg.set_ylabel('PSD (log scale, uV²/Hz)')
ax_eeg.set_title('EEG Alpha Band PSD (Welch Method)')

# Power plot setup
power_line, = ax_power.plot([], [], 'g-')
ax_power.set_xlabel('Time (s)')
ax_power.set_ylabel('Alpha Power (V²)')
ax_power.set_title('Alpha Power Over Time (8–12 Hz)')
ax_power.grid(True)

# Initialize animation
def init():
    raw_line.set_data([], [])
    psd_line.set_data([], [])
    power_line.set_data([], [])
    video_image.set_array(np.zeros((720, 960, 3), dtype=np.uint8))
    return raw_line, psd_line, power_line, video_image

# Update function
index = [0]
def update(frame):
    idx = index[0]
    
    # Update EEG data
    if idx + window_samples <= len(eeg_alpha):
        print(f"[EEG] Update index: {idx}, Window samples: {window_samples}")

        # Raw EEG plot
        time_window = np.arange(idx, idx + window_samples) / fsBionode
        raw_segment = raw_eeg[idx:idx + window_samples]
        raw_line.set_data(time_window, raw_segment)
        ax_raw.set_xlim(time_window[0], time_window[-1])
        ax_raw.set_ylim(np.min(raw_segment), np.max(raw_segment))

        # PSD calculation
        filtered_segment = eeg_alpha[idx:idx + window_samples] * np.hanning(window_samples)
        freqs, psd = signal.welch(
            filtered_segment,
            fs=fsBionode,
            window='hann',
            nperseg=window_samples,
            noverlap=window_samples//2,
            scaling='density'
        )
        
        # Update PSD plot
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        if np.any(alpha_mask):
            psd_line.set_data(freqs[alpha_mask], psd[alpha_mask])
            ax_eeg.set_xlim(8, 12)
            ax_eeg.set_ylim(np.maximum(1e-10, np.min(psd[alpha_mask])), 
                          np.max(psd[alpha_mask]))

            # Update power plot
            alpha_power = simps(psd[alpha_mask], freqs[alpha_mask])
            current_time = idx / fsBionode
            power_buffer.append(alpha_power)
            time_buffer.append(current_time)
            power_line.set_data(time_buffer, power_buffer)
            ax_power.set_xlim(max(0, current_time - window_sec), current_time)
            ax_power.set_ylim(1e-10, max(1e-9, max(power_buffer)))
        
        index[0] += step_samples
        print(f"[EEG] Alpha PSD freq range: {freqs[alpha_mask]}")
        print(f"[EEG] Alpha Power: {alpha_power:.4e}")

    
    # Update video
    if not queue_frame.empty():
        video_image.set_array(queue_frame.get())
    
    return raw_line, psd_line, power_line, video_image

# Start video thread
video_thread = Thread(target=run_video, daemon=True)
video_thread.start()

# Create animation
ani = animation.FuncAnimation(
    fig,
    update,
    init_func=init,
    frames=None,
    interval=step_sec * 1000,
    blit=True,
    cache_frame_data=False
)

plt.show()