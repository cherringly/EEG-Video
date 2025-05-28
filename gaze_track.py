import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import timedelta

class MediaPipeGazeTracking:
    def __init__(self):
        self.frame = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Eye landmarks indices (6 points per eye)
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]  # [p1, p2, p3, p4, p5, p6]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]  # [p1, p2, p3, p4, p5, p6]
        
        # Blink detection parameters
        self.ear_threshold = 0.25  # EAR threshold to consider eye as closed
        self.consec_frames = 3  # Number of consecutive frames below threshold to confirm blink
        self.frame_counter = 0  # Counter for consecutive frames with low EAR
        self.blink_counter = 0  # Total blink counter
        self.eye_state_history = []  # To store eye state and timestamps for CSV export
        self.last_recorded_time = -0.05  # Initialize to ensure first frame is recorded

    def refresh(self, frame):
        self.frame = frame
        self.landmarks = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0].landmark

    def _get_eye_points(self, indices):
        """Convert landmarks to numpy array of eye points"""
        h, w = self.frame.shape[:2]
        return np.array([(self.landmarks[i].x * w, self.landmarks[i].y * h) 
                        for i in indices])

    def _calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio for given eye landmarks"""
        # Extract the six points from the eye landmarks
        p1, p2, p3, p4, p5, p6 = eye_points
        
        # Calculate vertical distances
        A = np.linalg.norm(p2 - p6)  # |p2 - p6|
        B = np.linalg.norm(p3 - p5)  # |p3 - p5|
        
        # Calculate horizontal distance
        C = np.linalg.norm(p1 - p4)  # |p1 - p4|
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def is_blinking(self, current_time):
        """Detect blink using Eye Aspect Ratio (EAR)"""
        if not self.landmarks:
            if current_time >= self.last_recorded_time + 0.05:
                self.eye_state_history.append((current_time, "NO FACE", np.nan))
                self.last_recorded_time = current_time
            return False

        # Get eye landmarks
        left_eye = self._get_eye_points(self.left_eye_indices)
        right_eye = self._get_eye_points(self.right_eye_indices)
        
        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        
        # Average the EAR values for both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Determine eye state
        if ear < self.ear_threshold:
            eye_state = "CLOSED"
            self.frame_counter += 1
        else:
            eye_state = "OPEN"
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
            self.frame_counter = 0
        
        # Only record data every 0.05 seconds
        if current_time >= self.last_recorded_time + 0.05:
            self.eye_state_history.append((current_time, eye_state, ear))
            self.last_recorded_time = current_time
        
        return eye_state == "CLOSED" and self.frame_counter >= self.consec_frames

    def export_to_csv(self, filename="eye_state_log.csv"):
        """Export eye state history to CSV file"""
        df = pd.DataFrame(self.eye_state_history, columns=["Timestamp (s)", "Eye State", "EAR Value"])
        df.to_csv(filename, index=False)
        print(f"Eye state data exported to {filename}")

    def annotated_frame(self, current_time):
        if not self.landmarks:
            frame = self.frame.copy()
            cv2.putText(frame, "NO FACE DETECTED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        frame = self.frame.copy()
        h, w, _ = frame.shape
        
        # Draw eye landmarks
        for idx in self.left_eye_indices + self.right_eye_indices:
            pt = self.landmarks[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Calculate EAR if landmarks exist
        left_eye = self._get_eye_points(self.left_eye_indices)
        right_eye = self._get_eye_points(self.right_eye_indices)
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        # Format timestamp as seconds with 2 decimal places
        timestamp_str = f"{current_time:.2f} s"
        
        # Display information
        y_pos = 30
        cv2.putText(frame, f"Time: {timestamp_str}", (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        # cv2.putText(frame, f"EAR: {ear:.2f}", (50, y_pos), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # y_pos += 30
        # cv2.putText(frame, f"Blinks: {self.blink_counter}", (50, y_pos), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # y_pos += 30
        
        # Display eye state
        if ear < self.ear_threshold and self.frame_counter >= self.consec_frames:
            cv2.putText(frame, "EYES CLOSED", (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "EYES OPEN", (50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

if __name__ == "__main__":
    # Main execution
    cap = cv2.VideoCapture("video_recordings/alessandro.mov")
    gaze = MediaPipeGazeTracking()

    # Get video FPS to calculate timestamps
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate current time in seconds
        current_time = frame_count / fps
        frame_count += 1

        gaze.refresh(frame)
        frame = gaze.annotated_frame(current_time)

        # Update blink detection with current timestamp
        gaze.is_blinking(current_time)

        cv2.imshow("Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Export data to CSV when finished
    gaze.export_to_csv()


# emg_jaw_display.py

# import time
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from scipy.signal import butter, filtfilt
# from threading import Thread
# from queue import Queue, Full
# from bionodebinopen import fn_BionodeBinOpen
# from movement_track import HeadJawTracker

# # === CONFIG ===
# BLOCK_PATH = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
# VIDEO_PATH = r"video_recordings/alessandro_edit.mp4"
# ADC_RES = 12
# FS = 5537
# CHANNEL = 1
# WINDOW_SEC = 10
# EMG_YLIM = (-0.002, 0.002)
# JAW_BOX_SIZE = 100  # size of zoomed jaw crop

# paused = [False]
# pause_start = None
# t0_real = None
# queue_frame = Queue(maxsize=2)
# jaw_windows = []
# drawn_jaw_idx = 0

# # === Load & Filter EMG ===
# data = fn_BionodeBinOpen(BLOCK_PATH, ADC_RES, FS)
# raw = np.array(data['channelsData'])
# scale = 1.8 / 4096.0 / 10000  # V per unit / gain
# emg = (raw - 2048) * scale
# emg = np.nan_to_num(emg[CHANNEL])
# b, a = butter(4, 50 / (FS / 2), btype='low')
# emg_filtered = filtfilt(b, a, emg)
# time_arr = np.arange(len(emg_filtered)) / FS

# # === Video Thread ===
# def run_video():
#     global jaw_windows, t0_real
#     cap = cv2.VideoCapture(VIDEO_PATH)
#     start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#     tracker = HeadJawTracker()
#     moving_jaw = False
#     jaw_start = 0

#     while cap.isOpened():
#         if paused[0]:
#             cv2.waitKey(1)
#             continue

#         ret, frame = cap.read()
#         if not ret:
#             break

#         msec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
#         video_time = msec - start_time

#         if t0_real is None:
#             t0_real = time.time() - video_time
#         delay = t0_real + video_time - time.time()
#         if delay > 0:
#             time.sleep(delay)

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = tracker.face_mesh.process(rgb)

#         if results.multi_face_landmarks:
#             lm = results.multi_face_landmarks[0]
#             tracker.process(frame, lm)  # updates calibration and landmarks
#             annotated = tracker.annotated_frame(video_time)

#             if not moving_jaw and tracker.jaw_state != "Neutral":
#                 jaw_start = video_time
#                 moving_jaw = True
#             elif moving_jaw and tracker.jaw_state == "Neutral":
#                 jaw_windows.append((jaw_start, video_time))
#                 moving_jaw = False

#             h, w, _ = annotated.shape
#             lx, ly = int(lm.landmark[61].x * w), int(lm.landmark[61].y * h)
#             x1, x2 = max(lx - JAW_BOX_SIZE, 0), min(lx + JAW_BOX_SIZE, w)
#             y1, y2 = max(ly - JAW_BOX_SIZE, 0), min(ly + JAW_BOX_SIZE, h)
#             jaw_crop = annotated[y1:y2, x1:x2]
#             jaw_zoom = cv2.resize(jaw_crop, (240, 240))
#         else:
#             annotated = np.zeros((480, 640, 3), dtype=np.uint8)
#             jaw_zoom = np.zeros((240, 240, 3), dtype=np.uint8)

#         try:
#             queue_frame.put_nowait((annotated, jaw_zoom, video_time))
#         except Full:
#             pass

#     cap.release()

# Thread(target=run_video, daemon=True).start()

# # === Plot Setup ===
# fig = plt.figure(figsize=(10, 5))
# gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])

# ax_video = fig.add_subplot(gs[0, 0])
# ax_video.axis('off')
# img_disp = ax_video.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
# ax_video.set_title("Video Frame")

# ax_zoom = fig.add_subplot(gs[1, 0])
# ax_zoom.axis('off')
# img_zoom = ax_zoom.imshow(np.zeros((240, 240, 3), dtype=np.uint8))
# ax_zoom.set_title("Jaw Zoom")

# ax_emg = fig.add_subplot(gs[:, 1])
# line_emg, = ax_emg.plot([], [], lw=1, label="EMG")
# ax_emg.set_xlabel("Time (s)")
# ax_emg.set_ylabel("Voltage (V)")
# ax_emg.set_ylim(EMG_YLIM)
# ax_emg.grid(True)

# highlighted = []

# def init():
#     line_emg.set_data([], [])
#     return img_disp, img_zoom, line_emg

# def update(_):
#     global drawn_jaw_idx
#     if paused[0] or queue_frame.empty():
#         return img_disp, img_zoom, line_emg

#     frame, zoom, vtime = queue_frame.get()
#     img_disp.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     img_zoom.set_array(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))

#     t1 = vtime
#     t0 = max(0, t1 - WINDOW_SEC)
#     i0, i1 = int(t0 * FS), int(t1 * FS)
#     if i1 > len(emg_filtered): return img_disp, img_zoom, line_emg

#     t_win = time_arr[i0:i1]
#     y_win = emg_filtered[i0:i1]
#     line_emg.set_data(t_win, y_win)
#     ax_emg.set_xlim(t0, t1)

#     for span in highlighted:
#         span.remove()
#     highlighted.clear()

#     for start, end in jaw_windows[drawn_jaw_idx:]:
#         if start > t1: break
#         if end < t0: continue
#         shaded = ax_emg.axvspan(max(start, t0), min(end, t1), color='orange', alpha=0.3, label='Jaw Active')
#         highlighted.append(shaded)
#     drawn_jaw_idx = len(jaw_windows)

#     return img_disp, img_zoom, line_emg

# def on_key(event):
#     global pause_start, t0_real
#     if event.key == ' ':
#         paused[0] = not paused[0]
#         if paused[0]:
#             pause_start = time.time()
#         else:
#             t0_real += time.time() - pause_start

# def plot_static_first_minute():
#     fig, ax = plt.subplots(figsize=(12, 4))
#     i1 = int(60 * FS)
#     t = time_arr[:i1]
#     y = emg_filtered[:i1]
#     ax.plot(t, y, lw=1, label='EMG')
#     ax.set_xlim(0, 60)
#     ax.set_ylim(EMG_YLIM)
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel('Voltage (V)')
#     ax.set_title('First Minute EMG with Jaw Activity Highlighted')
#     ax.grid(True)

#     for start, end in jaw_windows:
#         if end < 0: continue
#         if start > 60: break
#         ax.axvspan(max(start, 0), min(end, 60), color='orange', alpha=0.3)

#     plt.tight_layout()
#     plt.show()

# fig.canvas.mpl_connect('key_press_event', on_key)
# ani = animation.FuncAnimation(fig, update, init_func=init, interval=20, blit=False)

# plt.tight_layout()
# plt.show()

# plot_static_first_minute()
