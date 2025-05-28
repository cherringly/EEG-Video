# # emg_movement_detector.py

# import numpy as np
# import cv2
# import mediapipe as mp
# from scipy import signal
# from bionodebinopen import fn_BionodeBinOpen
# from movement_track import HeadJawTracker


# def load_emg_data(block_path, adc_resolution, fs, channel):
#     data_dict = fn_BionodeBinOpen(block_path, adc_resolution, fs)
#     raw_data = np.array(data_dict['channelsData'])
#     scale_factor = 1.8 / 4096.0 / 10000
#     raw_data = (raw_data - 2048) * scale_factor
#     raw_data = np.nan_to_num(raw_data)
#     return raw_data[channel]


# def lowpass_filter_emg(data, fs):
#     sos = signal.butter(4, 50, btype='low', fs=fs, output='sos')
#     return signal.sosfiltfilt(sos, data)

# def format_time(seconds):
#     minutes = int(seconds // 60)
#     sec = int(seconds % 60)
#     millis = int((seconds - int(seconds)) * 1000)
#     return f"{minutes:02d}:{sec:02d}:{millis:03d}"


# def detect_emg_during_movement(emg_data, fs, movement_windows, type):
#     for start_sec, end_sec in movement_windows:
#         idx_start = int(start_sec * fs)
#         idx_end = int(end_sec * fs)
#         if idx_end > len(emg_data):
#             idx_end = len(emg_data)

#         segment = emg_data[idx_start:idx_end]
#         if len(segment) == 0:
#             continue

#         amplitude_mv = np.mean(np.abs(segment)) * 1000
#         start_str = format_time(start_sec)
#         end_str = format_time(end_sec)
#         print(f"Processing segment [{start_str} - {end_str}]: average amplitude is {amplitude_mv:.2f} mV")
#         if 1.0 <= amplitude_mv <= 3.0:
#             # print(f"EMG detected [{start_str} - {end_str}]: average amplitude is {amplitude_mv:.2f} mV")
#             print(f"EMG detected: {type}")

# def extract_movement_windows(video_path, fps):
#     cap = cv2.VideoCapture(video_path)
#     tracker = HeadJawTracker()
#     movement_windows = []
#     moving = False
#     start_time = 0
#     frame_idx = 0

#     with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             current_time = frame_idx / fps
#             frame_idx += 1
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(rgb)

#             if results.multi_face_landmarks:
#                 landmarks = results.multi_face_landmarks[0]
#                 _, pitch, yaw, _, jaw_state = tracker.process(frame, landmarks)

#                 if not moving and (abs(pitch) > 1 or abs(yaw) > 1):
#                     moving = True
#                     start_time = current_time
#                     type = "head"
                
#                 if not moving and (jaw_state != "Neutral"):
#                     moving = True
#                     start_time = current_time
#                     type = "jaw"

#                 elif moving and (abs(pitch) <= 3 and abs(yaw) <= 3 and jaw_state == "Neutral"):
#                     end_time = current_time
#                     movement_windows.append((start_time, end_time))
#                     moving = False

#     cap.release()
#     return movement_windows, type


# if __name__ == "__main__":
#     BLOCK_PATH = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
#     VIDEO_PATH = r"video_recordings/alessandro_edit.mp4"
#     FS = 5537
#     CHANNEL = 1
#     ADC_RES = 12

#     cap = cv2.VideoCapture(VIDEO_PATH)
#     FPS = cap.get(cv2.CAP_PROP_FPS)
#     cap.release()

#     emg_raw = load_emg_data(BLOCK_PATH, ADC_RES, FS, CHANNEL)
#     emg_filtered = lowpass_filter_emg(emg_raw, FS)

#     movement_windows, type = extract_movement_windows(VIDEO_PATH, FPS)
#     detect_emg_during_movement(emg_filtered, FS, movement_windows, type)







# emg_movement_detector.py
# emg_movement_detector.py

import numpy as np
import csv
import cv2
import mediapipe as mp
from scipy import signal
from bionodebinopen import fn_BionodeBinOpen
from movement_track import HeadJawTracker


def load_emg_data(block_path, adc_resolution, fs, channel):
    print("Loading EMG data...")
    data_dict = fn_BionodeBinOpen(block_path, adc_resolution, fs)
    raw_data = np.array(data_dict['channelsData'])
    scale_factor = 1.8 / 4096.0 / 10000
    raw_data = (raw_data - 2048) * scale_factor
    raw_data = np.nan_to_num(raw_data)
    print("EMG data loaded.")
    return raw_data[channel]


def lowpass_filter_emg(data, fs):
    print("Applying lowpass filter to EMG data...")
    sos = signal.butter(4, 50, btype='low', fs=fs, output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    print("Filtering complete.")
    return filtered


def format_time(seconds):
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes:02d}:{sec:02d}:{millis:03d}"


def detect_emg_during_movement(emg_data, fs, movement_windows, type):
    print(f"Detecting EMG over {len(movement_windows)} movement windows...")
    for i, (start_sec, end_sec) in enumerate(movement_windows):
        print(f"Processing window {i+1}: {start_sec:.2f}s to {end_sec:.2f}s")
        idx_start = int(start_sec * fs)
        idx_end = int(end_sec * fs)
        if idx_end > len(emg_data):
            idx_end = len(emg_data)

        segment = emg_data[idx_start:idx_end]
        if len(segment) == 0:
            print("  Empty segment, skipping.")
            continue

        amplitude_mv = np.mean(np.abs(segment)) * 1000
        print(f"  Avg amplitude: {amplitude_mv:.2f} mV")
        if 1.0 <= amplitude_mv <= 3.0:
            start_str = format_time(start_sec)
            end_str = format_time(end_sec)
            print(f"EMG detected [{start_str} - {end_str}]: type={type}")


def extract_movement_windows(video_path, fps):
    print("Extracting movement windows from video...")
    cap = cv2.VideoCapture(video_path)
    tracker = HeadJawTracker()
    head_windows = []
    jaw_windows = []
    moving_head = False
    moving_jaw = False
    head_start = 0
    jaw_start = 0
    frame_idx = 0

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_idx / fps
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                frame, pitch, yaw, _, jaw_state = tracker.process(frame, landmarks)
                # print(f"Frame {frame_idx}: Pitch={pitch:.2f}, Yaw={yaw:.2f}, Jaw={jaw_state}")

                # Head movement
                if not moving_head and (abs(pitch) > 0.85 or abs(yaw) > 0.85):
                    moving_head = True
                    head_start = current_time
                    print(f"  Head movement started at {head_start:.2f}s")

                elif moving_head and (abs(pitch) <= 0.85 and abs(yaw) <= 0.85):
                    head_end = current_time
                    head_windows.append((head_start, head_end))
                    print(f"  Head movement ended at {head_end:.2f}s")
                    moving_head = False

                # Jaw movement
                if not moving_jaw and (jaw_state != "Neutral"):
                    moving_jaw = True
                    jaw_start = current_time
                    print(f"  Jaw movement started at {jaw_start:.2f}s")

                elif moving_jaw and (jaw_state == "Neutral"):
                    jaw_end = current_time
                    jaw_windows.append((jaw_start, jaw_end))
                    print(f"  Jaw movement ended at {jaw_end:.2f}s")
                    moving_jaw = False

                cv2.imshow("Movement Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                cv2.putText(frame, "NO FACE DETECTED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Movement Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total head windows: {len(head_windows)}, jaw windows: {len(jaw_windows)}")
    return head_windows, jaw_windows



def export_movement_csv(head_windows, jaw_windows, fs, emg_data, output_path="movement_events.csv"):
    print(f"Exporting movement events to {output_path}...")

    def is_emg_detected(start_sec, end_sec):
        idx_start = int(start_sec * fs)
        idx_end = int(end_sec * fs)
        if idx_end > len(emg_data):
            idx_end = len(emg_data)
        segment = emg_data[idx_start:idx_end]
        if len(segment) == 0:
            return False
        amplitude_mv = np.mean(np.abs(segment)) * 1000
        return 1.0 <= amplitude_mv <= 3.0

    # Combine head and jaw events with tags
    all_windows = []
    for start, end in head_windows:
        all_windows.append((start, end, 'head'))
    for start, end in jaw_windows:
        all_windows.append((start, end, 'jaw'))

    # Sort by start time
    all_windows.sort(key=lambda x: x[0])

    # Create rows
    rows = []
    for start_sec, end_sec, movement_type in all_windows:
        time_str = format_time(start_sec)
        row = {'timestamp': time_str, 'head movement': '', 'jaw movement': '', 'emg detected': ''}

        if movement_type == 'head':
            row['head movement'] = 'yes'
        elif movement_type == 'jaw':
            row['jaw movement'] = 'yes'

        if is_emg_detected(start_sec, end_sec):
            row['emg detected'] = 'yes'

        rows.append(row)

    # Save to CSV
    with open(output_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['timestamp', 'head movement', 'jaw movement', 'emg detected'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV export complete: {output_path}")

if __name__ == "__main__":
    BLOCK_PATH = r"/Users/arundhatishankaran/Research/ear3.31.25_1.bin"
    VIDEO_PATH = r"video_recordings/alessandro.mov"
    FS = 5537
    CHANNEL = 1
    ADC_RES = 12

    print("Starting EMG movement detection...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Video FPS: {FPS}")

    emg_raw = load_emg_data(BLOCK_PATH, ADC_RES, FS, CHANNEL)
    emg_filtered = lowpass_filter_emg(emg_raw, FS)

    head_windows, jaw_windows = extract_movement_windows(VIDEO_PATH, FPS)
    
    detect_emg_during_movement(emg_filtered, FS, head_windows, type="head")
    detect_emg_during_movement(emg_filtered, FS, jaw_windows, type="jaw")
    export_movement_csv(head_windows, jaw_windows, FS, emg_filtered)



    print("Detection complete.")