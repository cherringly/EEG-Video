# eog_module.py

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from queue import Queue
from threading import Thread

from dataFunction import load_and_filter
from gazeFunction import gazeTrack

def eogDetect(filename, video_path, channel=1, ADCres=12, fsBionode=5537,
              eye_box_size=100, eeg_sync_offset=0.97, window_sec=10, ylim=(-0.0002, 0.0002)):

    paused = [False]
    t0_real = [None]
    prev_direction = [None]
    saccade_times = []
    eeg_queue = Queue(maxsize=2)
    pause_start = [None]
    vline_handles = []

    # ✅ Use new data loader
    data = load_and_filter(
        filename=filename,
        ADCres=ADCres,
        sampR=fsBionode,
        filter_config={channel: {"type": "low", "cutoff": 50}}  # Can customize
    )

    filtered = data["filtered"][channel]
    time_arr = data["time"]

    # ✅ Launch gaze thread
    Thread(target=gazeTrack, args=(video_path, eeg_queue, paused, saccade_times,
                                   prev_direction, t0_real, eye_box_size), daemon=True).start()

    # === Plot Setup ===
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1], wspace=0.3, hspace=0.2)
    ax_main = fig.add_subplot(gs[0, 0])
    im_main = ax_main.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
    ax_main.axis('off')
    ax_main.set_title("Main Video")

    ax_zoom = fig.add_subplot(gs[1, 0])
    im_zoom = ax_zoom.imshow(np.zeros((240, 240, 3), dtype=np.uint8))
    ax_zoom.axis('off')
    ax_zoom.set_title("Eyes Zoom")

    ax_eeg = fig.add_subplot(gs[:, 1])
    line, = ax_eeg.plot([], [], lw=1, label="EEG signal")
    peak_marker, = ax_eeg.plot([], [], 'rx', label="Peak")
    trough_marker, = ax_eeg.plot([], [], 'bx', label="Trough")
    ax_eeg.set_xlabel("Time (s)")
    ax_eeg.set_ylabel("Voltage (V)")
    ax_eeg.set_xlim(0, window_sec)
    ax_eeg.set_ylim(ylim)
    ax_eeg.grid(True)
    ax_eeg.legend(loc='upper right')

    def init():
        line.set_data([], [])
        peak_marker.set_data([], [])
        trough_marker.set_data([], [])
        return im_main, im_zoom, line, peak_marker, trough_marker

    def update(_):
        nonlocal vline_handles
        if paused[0] or eeg_queue.empty():
            return im_main, im_zoom, line, peak_marker, trough_marker
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

            for vline in vline_handles:
                vline.remove()
            vline_handles = []

            peak_times, trough_times, peak_vals, trough_vals = [], [], [], []
            pairs = [(saccade_times[i], saccade_times[i+1]) for i in range(0, len(saccade_times)-1, 2)]
            for start, end in pairs:
                eeg_start = int((start - eeg_sync_offset) * fsBionode)
                eeg_end = int((end - eeg_sync_offset) * fsBionode)
                if eeg_start < 0 or eeg_end > len(filtered): continue
                eeg_snippet = filtered[eeg_start:eeg_end]
                snippet_times = time_arr[eeg_start:eeg_end]
                if len(eeg_snippet) > 0:
                    peak_idx = np.argmax(eeg_snippet)
                    trough_idx = np.argmin(eeg_snippet)
                    peak_val = eeg_snippet[peak_idx]
                    trough_val = eeg_snippet[trough_idx]
                    if abs(peak_val - trough_val) >= 0.0001:
                        peak_times.append(snippet_times[peak_idx])
                        peak_vals.append(peak_val)
                        trough_times.append(snippet_times[trough_idx])
                        trough_vals.append(trough_val)
                vline_handles += [
                    ax_eeg.axvline(start - eeg_sync_offset, color='magenta', linestyle='--', alpha=0.7, linewidth=1),
                    ax_eeg.axvline(end - eeg_sync_offset, color='magenta', linestyle='--', alpha=0.7, linewidth=1)
                ]
            peak_marker.set_data(peak_times, peak_vals)
            trough_marker.set_data(trough_times, trough_vals)
        return im_main, im_zoom, line, peak_marker, trough_marker

    def on_key(event):
        if event.key == ' ':
            if not paused[0]:
                pause_start[0] = time.time()
                paused[0] = True
            else:
                paused_duration = time.time() - pause_start[0]
                t0_real[0] += paused_duration
                paused[0] = False

    fig.canvas.mpl_connect('key_press_event', on_key)
    ani = animation.FuncAnimation(fig, update, init_func=init, interval=20, blit=False)
    plt.tight_layout()
    plt.show()
