import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
import cv2
from threading import Thread
from queue import Queue, Full
from gaze_track_archis import MediaPipeGazeTracking
from bionodebinopen import fn_BionodeBinOpen
import mediapipe as mp
from paralelll import bandpass_filter_alpha, compute_alpha_power, smooth_alpha_power
from emg_movement_detector import extract_movement_windows, detect_emg_during_movement, export_movement_csv

# === CONFIG ===
filename        = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
ADCres          = 12
fsBionode       = 5537
channel         = 1
video_path      = r"\Users\maryz\EEG-Video\video_recordings\alessandro_edit.mp4"
eeg_sync_offset = 0.97
window_sec      = 20      # 20 s sliding window for both EEG & alpha
step_sec        = 0.02    # 20 ms update interval
eye_box_size    = 100     # half-width of dynamic eye crop
ylim_eeg        = (-0.0002, 0.0002)

# === STATE ===
paused         = [False]
pause_start    = None
t0_real        = None
queue_frame    = Queue(maxsize=2)
saccade_times  = []
prev_direction = None
vline_handles  = []

# === EMG PROCESSING STATE ===
emg_processing_complete = [False]
head_windows = []
jaw_windows = []
emg_thread_started = [False]

# === LOAD & FILTER RAW EEG ONCE ===
data       = fn_BionodeBinOpen(filename, ADCres, fsBionode)
raw_all    = data["channelsData"].astype(np.float32)
raw_all    = (raw_all - 2**11) * 1.8 / (2**12 * 1000)
raw_chan   = raw_all[channel]
b, a       = butter(4, 50/(fsBionode/2), btype='low')
filtered   = filtfilt(b, a, raw_chan)
time_arr   = np.arange(len(filtered)) / fsBionode

# === ALPHA POWER (1-second epochs) ===
eeg_alpha       = bandpass_filter_alpha(raw_chan, fsBionode)
time_min, alpha_power = compute_alpha_power(eeg_alpha, fsBionode, 1)
smoothed_power  = smooth_alpha_power(alpha_power, fsBionode, 1)
time_sec_alpha  = time_min * 60  

# === MEDIAPIPE FACE MESH ===
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)

def run_emg_processing():
    """Run EMG movement detection in background thread"""
    global head_windows, jaw_windows, emg_processing_complete
    
    print("=== EMG PROCESSING THREAD STARTED ===")
    print(f"Video path: {video_path}")
    
    # Get video FPS for EMG processing
    try:
        cap_emg = cv2.VideoCapture(video_path)
        fps_emg = cap_emg.get(cv2.CAP_PROP_FPS)
        cap_emg.release()
        print(f"Video FPS: {fps_emg}")
    except Exception as e:
        print(f"Error getting video FPS: {e}")
        emg_processing_complete[0] = True
        return
    
    try:
        print("Extracting movement windows...")
        # Extract movement windows - but we need to modify this to not show GUI
        # We'll call a headless version or modify the function
        
        # Try to import and call the function with a headless flag if available
        try:
            from emg_movement_detector import extract_movement_windows_headless
            head_windows, jaw_windows = extract_movement_windows_headless(video_path, fps_emg)
        except ImportError:
            # If headless version doesn't exist, we need to work around the cv2.imshow issue
            print("WARNING: extract_movement_windows_headless not found. Trying to monkey-patch cv2.imshow...")
            
            # Temporarily disable cv2.imshow by replacing it with a no-op function
            original_imshow = cv2.imshow
            original_waitkey = cv2.waitKey
            original_destroyallwindows = cv2.destroyAllWindows
            
            def dummy_imshow(*args, **kwargs):
                pass  # Do nothing instead of showing window
            
            def dummy_waitkey(*args, **kwargs):
                return ord('q')  # Always return 'q' to continue processing
            
            def dummy_destroyallwindows(*args, **kwargs):
                pass  # Do nothing
            
            # Replace OpenCV GUI functions with dummy versions
            cv2.imshow = dummy_imshow
            cv2.waitKey = dummy_waitkey  
            cv2.destroyAllWindows = dummy_destroyallwindows
            
            try:
                head_windows, jaw_windows = extract_movement_windows(video_path, fps_emg)
            finally:
                # Restore original functions
                cv2.imshow = original_imshow
                cv2.waitKey = original_waitkey
                cv2.destroyAllWindows = original_destroyallwindows
        
        print(f"Found {len(head_windows)} head windows, {len(jaw_windows)} jaw windows")
        
        print("Detecting EMG during head movements...")
        # Detect EMG during movement - this will print EMG detection messages
        detect_emg_during_movement(filtered, fsBionode, head_windows, type="head")
        
        print("Detecting EMG during jaw movements...")
        detect_emg_during_movement(filtered, fsBionode, jaw_windows, type="jaw")
        
        print("Exporting movement CSV...")
        # Export CSV
        export_movement_csv(head_windows, jaw_windows, fsBionode, filtered)
        
        print("=== EMG PROCESSING COMPLETED SUCCESSFULLY! ===")
        
    except Exception as e:
        print(f"=== ERROR IN EMG PROCESSING: {e} ===")
        import traceback
        traceback.print_exc()
    
    emg_processing_complete[0] = True

def run_video():
    global prev_direction, saccade_times, t0_real, emg_thread_started
    cap    = cv2.VideoCapture(video_path)
    start  = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    gaze   = MediaPipeGazeTracking()
    
    # Start EMG processing thread once video starts
    if not emg_thread_started[0]:
        print("Starting EMG processing thread...")
        emg_thread = Thread(target=run_emg_processing, daemon=True)
        emg_thread.start()
        emg_thread_started[0] = True
        print("EMG thread started!")
    
    while True:
        if paused[0]:
            time.sleep(0.1)
            continue
        ret, frame = cap.read()
        if not ret:
            break

        # synchronize to real time
        msec     = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        vid_time = msec - start
        if t0_real is None:
            t0_real = time.time() - vid_time
        dt = t0_real + vid_time - time.time()
        if dt > 0:
            time.sleep(dt)

        # gaze tracking & annotation
        gaze.refresh(frame)
        annotated = gaze.annotated_frame(vid_time)

        # saccade detection on gaze direction change
        d = gaze.gaze_direction
        if prev_direction is not None and d != prev_direction:
            saccade_times.append(vid_time)
        prev_direction = d

        # enqueue for animation
        try:
            queue_frame.put_nowait((annotated, vid_time))
        except Full:
            pass

    cap.release()
    gaze.export_to_csv()
    gaze.export_gaze_to_csv()
    gaze.export_saccades()

Thread(target=run_video, daemon=True).start()

# === FIGURE SETUP ===
fig = plt.figure(figsize=(12, 7))
gs  = fig.add_gridspec(3, 2,
                       width_ratios=[1, 1.5],
                       height_ratios=[1, 1, 1],
                       wspace=0.3, hspace=0.3)

# main video
ax_main = fig.add_subplot(gs[0, 0])
im_main = ax_main.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
ax_main.axis('off')
ax_main.set_title("Video Feed")

# dynamic eyes zoom
ax_zoom = fig.add_subplot(gs[1, 0])
im_zoom = ax_zoom.imshow(np.zeros((2*eye_box_size, 2*eye_box_size, 3), dtype=np.uint8))
ax_zoom.axis('off')
ax_zoom.set_title("Eyes Zoom")

# EEG plot spans top two rows on right
ax_eeg = fig.add_subplot(gs[0:2, 1])
line_eeg, = ax_eeg.plot([], [], lw=1, label="EEG signal")
peak_marker, = ax_eeg.plot([], [], 'rx', label="Peak")
trough_marker, = ax_eeg.plot([], [], 'bx', label="Trough")
# Add invisible line for legend entry
eog_event_line, = ax_eeg.plot([], [], color='magenta', linestyle='--', alpha=0.7, linewidth=1, label="Detected EOG event")
ax_eeg.set_xlabel("Time (s)")
ax_eeg.set_ylabel("Voltage (V)")
ax_eeg.set_xlim(0, window_sec)
ax_eeg.set_ylim(*ylim_eeg)
ax_eeg.grid(True)
ax_eeg.legend(loc='upper right')

# Alpha power plot (bottom full width) – y‐limits from actual data
ax_alpha = fig.add_subplot(gs[2, :])
raw_line,    = ax_alpha.plot([], [], color='green', alpha=0.5, label="Alpha (raw)")
smooth_line, = ax_alpha.plot([], [], color='red', linewidth=2, label="Alpha (smoothed)")
ax_alpha.set_xlabel("Time (s)")
ax_alpha.set_ylabel("Alpha Power (V²)")
ax_alpha.set_yscale('log')
ax_alpha.set_xlim(0, window_sec)
ax_alpha.set_ylim(np.min(alpha_power), np.max(alpha_power))
ax_alpha.grid(True)
ax_alpha.legend()

def init():
    im_main.set_array(np.zeros_like(im_main.get_array()))
    im_zoom.set_array(np.zeros_like(im_zoom.get_array()))
    line_eeg.set_data([], [])
    peak_marker.set_data([], [])
    trough_marker.set_data([], [])
    raw_line.set_data([], [])
    smooth_line.set_data([], [])
    return [im_main, im_zoom, line_eeg, peak_marker, trough_marker, raw_line, smooth_line]

def update(_):
    global vline_handles
    if paused[0] or queue_frame.empty():
        return [im_main, im_zoom, line_eeg, peak_marker, trough_marker, raw_line, smooth_line]

    annotated, tv = queue_frame.get()
    rgb           = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    im_main.set_array(rgb)

    # dynamic eye crop using FaceMesh landmarks
    res = face_mesh.process(rgb)
    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        h, w, _ = rgb.shape
        lx, ly = int(lm[33].x * w), int(lm[33].y * h)
        rx, ry = int(lm[263].x * w), int(lm[263].y * h)
        cx, cy = (lx + rx)//2, (ly + ry)//2
        x1, x2 = max(cx - eye_box_size, 0), min(cx + eye_box_size, w)
        y1, y2 = max(cy - eye_box_size, 0), min(cy + eye_box_size, h)
        crop = rgb[y1:y2, x1:x2]
        zoom = cv2.resize(crop, (2*eye_box_size, 2*eye_box_size))
    else:
        zoom = np.zeros((2*eye_box_size, 2*eye_box_size, 3), dtype=np.uint8)
    im_zoom.set_array(zoom)

    # EEG sliding window & peak/trough detection
    te = tv - eeg_sync_offset
    t0 = max(0, te - window_sec)
    i0, i1 = int(t0 * fsBionode), int(te * fsBionode)
    if i1 <= len(filtered):
        t_win = time_arr[i0:i1]
        y_win = filtered[i0:i1]
        line_eeg.set_data(t_win, y_win)
        ax_eeg.set_xlim(t0, t0 + window_sec)

        peak_times, trough_times, peak_vals, trough_vals = [], [], [], []

        # Remove old pink lines
        for vline in vline_handles:
            vline.remove()
        vline_handles = []

        # Use saccade pairs as window
        pairs = [(saccade_times[i], saccade_times[i+1]) for i in range(0, len(saccade_times)-1, 2)]
        for start, end in pairs:
            eeg_start = int((start - eeg_sync_offset) * fsBionode)
            eeg_end = int((end - eeg_sync_offset) * fsBionode)
            if eeg_start < 0 or eeg_end > len(filtered):
                continue
            eeg_snippet = filtered[eeg_start:eeg_end]
            snippet_times = time_arr[eeg_start:eeg_end]

            if len(eeg_snippet) > 0:
                peak_idx = np.argmax(eeg_snippet)
                trough_idx = np.argmin(eeg_snippet)

                peak_times.append(snippet_times[peak_idx])
                peak_vals.append(eeg_snippet[peak_idx])
                trough_times.append(snippet_times[trough_idx])
                trough_vals.append(eeg_snippet[trough_idx])

            # Pink lines for window
            vline1 = ax_eeg.axvline(start - eeg_sync_offset, color='magenta', linestyle='--', alpha=0.7, linewidth=1)
            vline2 = ax_eeg.axvline(end - eeg_sync_offset, color='magenta', linestyle='--', alpha=0.7, linewidth=1)
            vline_handles.extend([vline1, vline2])

        peak_marker.set_data(peak_times, peak_vals)
        trough_marker.set_data(trough_times, trough_vals)

    # Alpha power sliding window (build in first window_sec then scroll)
    if te < window_sec:
        mask = (time_sec_alpha >= 0) & (time_sec_alpha <= te)
        rel_t = time_sec_alpha[mask]
    else:
        mask = (time_sec_alpha >= t0) & (time_sec_alpha <= t0 + window_sec)
        rel_t = time_sec_alpha[mask] - t0
    if np.any(mask):
        raw_line.set_data(rel_t,           alpha_power[mask])
        smooth_line.set_data(rel_t, smoothed_power[mask])
        ax_alpha.set_xlim(0, window_sec)

    return [im_main, im_zoom, line_eeg, peak_marker, trough_marker, raw_line, smooth_line]

def on_key(event):
    global pause_start, t0_real
    if event.key == ' ':
        if not paused[0]:
            pause_start = time.time()
            paused[0]   = True
            print("Paused")
        else:
            paused_duration = time.time() - pause_start
            t0_real       += paused_duration
            paused[0]      = False
            print("Resumed")

fig.canvas.mpl_connect('key_press_event', on_key)

ani = animation.FuncAnimation(
    fig, update, init_func=init,
    interval=step_sec*1000,
    blit=False, cache_frame_data=False
)

print("Starting GUI... EMG processing will run in background and print results as they come.")
plt.tight_layout()
plt.show()

# Check if EMG processing completed after GUI closes
if emg_processing_complete[0]:
    print("All processing completed successfully!")
else:
    print("EMG processing may still be running in background...")