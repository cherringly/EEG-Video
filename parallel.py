import numpy as np
from offline_power import load_and_preprocess_data, bandpass_filter_alpha, compute_alpha_power
from movement_track import MediaPipeGazeTracking
import pandas as pd
import cv2
import mediapipe as mp

def compute_minute_alpha_stats(time_minutes, alpha_powers):
    total_minutes = int(time_minutes[-1]) + 1
    minute_powers = []
    for m in range(total_minutes):
        idx = (time_minutes >= m) & (time_minutes < m + 1)
        if np.any(idx):
            avg_power = np.mean(alpha_powers[idx])
        else:
            avg_power = np.nan
        minute_powers.append(avg_power)
    return np.array(minute_powers)

def compute_eye_state_vector_from_gaze_tracker(video_path, eye_csv_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    gaze_tracker = MediaPipeGazeTracking()

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps
            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                gaze_tracker.analyze(frame, landmarks.landmark, current_time)

    cap.release()
    gaze_tracker.export_to_csv(eye_csv_path)


def compute_eye_state_vector(csv_path, window_sec):
    df = pd.read_csv(csv_path)
    duration = df['Timestamp (s)'].max()
    n_windows = int(duration // window_sec)
    eye_states = np.zeros(n_windows, dtype=int)

    for i in range(n_windows):
        start = i * window_sec
        end = start + window_sec
        window_data = df[(df['Timestamp (s)'] >= start) & (df['Timestamp (s)'] < end)]
        closed_ratio = np.mean(window_data['Eye State'] == 'CLOSED') if not window_data.empty else 0
        eye_states[i] = 0 if closed_ratio > 0.5 else 1
    return eye_states

def classify_minute_eye_state(eye_states, window_sec):
    windows_per_minute = int(60 / window_sec)
    n_minutes = len(eye_states) // windows_per_minute
    minute_labels = []
    for m in range(n_minutes):
        start = m * windows_per_minute
        end = start + windows_per_minute
        minute_segment = eye_states[start:end]
        closed_ratio = np.mean(minute_segment == 0)
        label = 'Closed' if closed_ratio > 0.4 else 'Open'
        minute_labels.append(label)
        print(f"Minute {m}: {label} (Closed Ratio: {closed_ratio:.2f})")
    return minute_labels

def detect_eeg(minute_eye_states, minute_alpha_powers):
    results = []
    for i in range(0, min(len(minute_eye_states), len(minute_alpha_powers)) - 1, 2):
        cond_eye = (minute_eye_states[i] == 'Open') and (minute_eye_states[i+1] == 'Closed')
        cond_power = (minute_alpha_powers[i] < minute_alpha_powers[i+1])
        eeg_detected = cond_eye and cond_power
        results.append(eeg_detected)
        print(f"EEG DETECTED between minute {i} and {i+1}: {eeg_detected}")
    return results

def run_eeg_detection():
    channel = 1
    ADCres = 12
    fs = 5537
    blockPath = r"\Users\maryz\EEG-Video\bin_files\ear3.31.25_1.bin"
    video_path = "video_recordings/alessandro.mov"
    eye_csv_path = "eye_state_log.csv"
    window_sec = 1

    compute_eye_state_vector_from_gaze_tracker(video_path, eye_csv_path)

    raw = load_and_preprocess_data(blockPath, ADCres, fs, channel)
    filtered = bandpass_filter_alpha(raw, fs)
    time_min, alpha_powers = compute_alpha_power(filtered, fs, window_sec)
    minute_alpha_powers = compute_minute_alpha_stats(time_min, alpha_powers)

    eye_states_1s = compute_eye_state_vector(eye_csv_path, window_sec)
    minute_eye_states = classify_minute_eye_state(eye_states_1s, window_sec)

    print("Minute Eye States:", minute_eye_states)
    print("Minute Alpha Powers:", minute_alpha_powers)

    detect_eeg(minute_eye_states, minute_alpha_powers)

if __name__ == "__main__":
    run_eeg_detection()