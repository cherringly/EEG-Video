import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
import cv2
from threading import Thread
from queue import Queue, Full
from gaze_track import MediaPipeGazeTracking
from bionodebinopen import fn_BionodeBinOpen
import mediapipe as mp

# === CONFIG ===
filename        = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
ADCres          = 12
fsBionode       = 5537
channel         = 1
video_path      = r"\Users\maryz\EEG-Video\video_recordings\alessandro_edit.mp4"
eeg_sync_offset = 0.97
window_sec      = 10
ylim            = (-0.0002, 0.0002)
eye_box_size    = 100  # half‐width of eye crop around center

paused          = [False]
pause_start     = None
t0_real         = None
eeg_queue       = Queue(maxsize=2)
saccade_times   = []
prev_direction  = None
drawn_sacc_idx  = 0    # counts how many pairs drawn

# load & filter EEG
data      = fn_BionodeBinOpen(filename, ADCres, fsBionode)
rawCha    = data["channelsData"].astype(np.float32)
rawCha    = (rawCha - 2**11) * 1.8 / (2**12 * 1000)
b, a      = butter(4, 50/(fsBionode/2), btype='low')
filtered  = filtfilt(b, a, rawCha[channel])
time_arr  = np.arange(len(filtered)) / fsBionode

# setup MediaPipe face mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)

def run_video():
    global prev_direction, saccade_times, t0_real
    cap = cv2.VideoCapture(video_path)
    start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    gaze = MediaPipeGazeTracking()
    while True:
        if paused[0]:
            time.sleep(0.1)
            continue
        ret, frame = cap.read()
        if not ret:
            break
        msec     = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        vid_time = msec - start_time

        # real-time sync (initialize once)
        if t0_real is None:
            t0_real = time.time() - vid_time
        to_sleep = t0_real + vid_time - time.time()
        if to_sleep > 0:
            time.sleep(to_sleep)

        # gaze & annotate
        gaze.refresh(frame)
        annotated = gaze.annotated_frame(vid_time)

        # main video small
        main_disp = cv2.resize(annotated, (320, 240))

        # face mesh for tighter eye zoom
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            lx, ly = int(lm[33].x * w), int(lm[33].y * h)
            rx, ry = int(lm[263].x * w), int(lm[263].y * h)
            cx, cy = (lx + rx)//2, (ly + ry)//2
            x1, x2 = max(cx - eye_box_size, 0), min(cx + eye_box_size, w)
            y1, y2 = max(cy - eye_box_size, 0), min(cy + eye_box_size, h)
            cropped = annotated[y1:y2, x1:x2]
            zoom_disp = cv2.resize(cropped, (240, 240))
        else:
            zoom_disp = np.zeros((240, 240, 3), dtype=np.uint8)

        # saccade detection
        d = gaze.gaze_direction
        if prev_direction and d != prev_direction:
            saccade_times.append(vid_time)
        prev_direction = d

        try:
            eeg_queue.put_nowait((main_disp, zoom_disp, vid_time))
        except Full:
            pass

    cap.release()
    gaze.export_to_csv()
    gaze.export_gaze_to_csv()
    gaze.export_saccades()

Thread(target=run_video, daemon=True).start()

# === PLOT SETUP ===
fig = plt.figure(figsize=(10, 5))
gs  = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1], wspace=0.3, hspace=0.2)

ax_main = fig.add_subplot(gs[0, 0])
im_main = ax_main.imshow(np.zeros((240,320,3), dtype=np.uint8))
ax_main.axis('off')
ax_main.set_title("Main Video")

ax_zoom = fig.add_subplot(gs[1, 0])
im_zoom = ax_zoom.imshow(np.zeros((240,240,3), dtype=np.uint8))
ax_zoom.axis('off')
ax_zoom.set_title("Eyes Zoom")

ax_eeg = fig.add_subplot(gs[:, 1])
line, = ax_eeg.plot([], [], lw=1, label="EOG signal")
sac,  = ax_eeg.plot([], [], 'rx', label="EOG event")
ax_eeg.set_xlabel("Time (s)")
ax_eeg.set_ylabel("Voltage (V)")
ax_eeg.set_xlim(0, window_sec)
ax_eeg.set_ylim(ylim)
ax_eeg.grid(True)
ax_eeg.legend(loc='upper right')

def init():
    line.set_data([], [])
    sac.set_data([], [])
    return im_main, im_zoom, line, sac

def update(_):
    global drawn_sacc_idx
    if paused[0] or eeg_queue.empty():
        return im_main, im_zoom, line, sac

    main_img, zoom_img, tv = eeg_queue.get()
    im_main.set_array(cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB))
    im_zoom.set_array(cv2.cvtColor(zoom_img, cv2.COLOR_BGR2RGB))

    te = tv - eeg_sync_offset
    t0 = max(0, te - window_sec)
    i0, i1 = int(t0 * fsBionode), int(te * fsBionode)
    if i1 <= len(filtered):
        t_win = time_arr[i0:i1]
        y_win = filtered[i0:i1]
        line.set_data(t_win, y_win)
        ax_eeg.set_xlim(t0, t0 + window_sec)

        pairs = [(saccade_times[i], saccade_times[i+1])
                 for i in range(0, len(saccade_times)-1, 2)]

        pts, vals = [], []
        for start, end in pairs:
            for s in (start, end):
                se = s - eeg_sync_offset
                if t0 <= se <= te:
                    idx = int(se * fsBionode)
                    pts.append(time_arr[idx])
                    vals.append(filtered[idx])
        sac.set_data(pts, vals)

        for pi in range(drawn_sacc_idx, len(pairs)):
            st, en = pairs[pi]
            ax_eeg.axvline(st - eeg_sync_offset, color='magenta', linestyle='--', alpha=0.7, linewidth=1)
            ax_eeg.axvline(en - eeg_sync_offset, color='magenta', linestyle='--', alpha=0.7, linewidth=1)
        drawn_sacc_idx = len(pairs)

    return im_main, im_zoom, line, sac

def on_key(event):
    global pause_start, t0_real
    if event.key == ' ':
        if not paused[0]:
            pause_start = time.time()
            paused[0] = True
            print("Paused")
        else:
            paused_duration = time.time() - pause_start
            t0_real       += paused_duration
            paused[0]      = False
            print("Resumed")

fig.canvas.mpl_connect('key_press_event', on_key)
ani = animation.FuncAnimation(fig, update, init_func=init, interval=20, blit=False)

plt.tight_layout()
plt.show()