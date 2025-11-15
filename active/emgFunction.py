# emg_module.py
import numpy as np
import csv
import cv2
import mediapipe as mp
from scipy import signal
from queue import Queue
from threading import Thread
from datetime import datetime
from active.movement_track import HeadJawTracker


def emgDetect(block_path, video_path, channel=1, ADCres=12, fsBionode=5537,
              lowpass_cutoff=50, emg_threshold=(1.0, 3.0), eye_box_size=100,
              output_csv="movement_events.csv"):

    # === Helper Functions ===
        
    def fn_BionodeBinOpen(packedFileDir: str, ADCres: int, sampR: int) -> dict:
        """
        Open a Bionode-generated .bin file, parse header and unpack 12-bit ADC samples
        into a channels×time array.

        Args:
            packedFileDir: path to the .bin file
            ADCres: ADC resolution in bits (e.g. 24)
            sampR: sampling rate (Hz) for the time vector

        Returns:
            dict with keys:
            - 'Date': datetime object from file header
            - 'sampleRate': sample rate from header (int)
            - 'numChannels': number of channels (int)
            - 'channelsData': ndarray, shape (num_channels, total_samples)
            - 'time': ndarray, shape (total_samples,)
        """
        # Read entire file as uint8
        try:
            with open(packedFileDir, 'rb') as f:
                raw_bytes = f.read()
        except IOError as e:
            raise IOError(f"Could not open file {packedFileDir}: {e}")

        raw_data   = np.frombuffer(raw_bytes, dtype=np.uint8)
        packet_len = 58
        packet_num = len(raw_data) // packet_len
        if packet_num < 2:
            raise ValueError("File contains no data packets.")

        # --- Parse header packet (first 58 bytes) ---
        hdr = raw_data[:packet_len]
        year        = (hdr[0] << 8) + hdr[1]
        month       = int(hdr[2])
        day         = int(hdr[3])
        hour        = int(hdr[4])
        minute      = int(hdr[5])
        second      = int(hdr[6])
        sampleRate  = (hdr[7] << 8) + hdr[8]
        numChannels = int(hdr[9])
        file_date   = datetime(year, month, day, hour, minute, second)

        # Determine how many 12-bit samples per channel per packet
        n_samps_per_pkt = 24 // numChannels
        n_data_pkts     = packet_num - 1
        total_samps     = n_data_pkts * n_samps_per_pkt

        # Preallocate channels × time array
        channelsData = np.zeros((numChannels, total_samps), dtype=np.uint16)

        print("Unpacking Data from file...")

        # Loop over data packets
        for i in range(1, packet_num):
            start = i * packet_len
            end   = start + packet_len
            pkt   = raw_data[start:end]

            # Extract the 3-byte groups: bytes 8 through (58-15)  → indices [7:43]
            x = pkt[7:packet_len-15]
            x = x.reshape(-1, 3)

            # Combine into 24-bit words
            y = (
                (x[:, 0].astype(np.uint32) << 16) |
                (x[:, 1].astype(np.uint32) <<  8) |
                x[:, 2].astype(np.uint32)
            )

            # Split each into two 12-bit samples
            high = (y & 0xFFF000) >> 12
            low  =  y & 0x000FFF
            interleaved = np.column_stack((high, low)).ravel(order='F')

            # Reshape into (channels × samples_per_chunk)
            pkt_data = interleaved.reshape((numChannels, n_samps_per_pkt), order='F')

            idx0 = (i-1) * n_samps_per_pkt
            idx1 = idx0 + n_samps_per_pkt
            channelsData[:, idx0:idx1] = pkt_data

            # Progress update every ~5%
            if i % max(1, round(packet_num * 0.05)) == 0:
                print(f"Progress: {i/packet_num*100:.0f}%")

        print("Unpacking Completed!")

        # Build the time vector
        time = np.arange(total_samps) / sampR

        return {
            'Date':         file_date,
            'sampleRate':   sampleRate,
            'numChannels':  numChannels,
            'channelsData': channelsData,
            'time':         time
        }

    def format_time(seconds):
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{minutes:02d}:{sec:02d}:{millis:03d}"

    def load_emg():
        print("Loading EMG data...")
        data_dict = fn_BionodeBinOpen(block_path, ADCres, fsBionode)
        raw = np.array(data_dict['channelsData'])[channel]
        if ADCres == 12:
            scaled = (raw - 2**11) * 1.8 / (2**12 * 10000)
        elif ADCres == 24:
            scaled = (raw - 2**23) * 1.8 / (2**24 * 10000)
        else:
            raise ValueError(f"Unsupported ADC resolution: {ADCres}")
        print("EMG data loaded.")
        return np.nan_to_num(scaled)

    def lowpass_filter(data):
        print(f"Applying {lowpass_cutoff}Hz lowpass filter...")
        sos = signal.butter(4, lowpass_cutoff, btype='low', fs=fsBionode, output='sos')
        filtered = signal.sosfiltfilt(sos, data)
        print("Filtering complete.")
        return filtered

    def extract_movement_windows():
        print("Extracting movement windows from video...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        tracker = HeadJawTracker()
        head_windows, jaw_windows = [], []
        moving_head, moving_jaw = False, False
        head_start, jaw_start = 0, 0
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

                    # Head movement
                    if not moving_head and (abs(pitch) > 0.85 or abs(yaw) > 0.85):
                        moving_head = True
                        head_start = current_time
                    elif moving_head and (abs(pitch) <= 0.85 and abs(yaw) <= 0.85):
                        head_windows.append((head_start, current_time))
                        moving_head = False

                    # Jaw movement
                    if not moving_jaw and (jaw_state != "Neutral"):
                        moving_jaw = True
                        jaw_start = current_time
                    elif moving_jaw and (jaw_state == "Neutral"):
                        jaw_windows.append((jaw_start, current_time))
                        moving_jaw = False

                cv2.imshow("Movement Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Total head windows: {len(head_windows)}, jaw windows: {len(jaw_windows)}")
        return head_windows, jaw_windows, fps

    def detect_emg(mov_windows, movement_type):
        print(f"Detecting EMG over {len(mov_windows)} {movement_type} windows...")
        detected_windows = []
        for start, end in mov_windows:
            idx_start = int(start * fsBionode)
            idx_end = int(end * fsBionode)
            segment = emg_filtered[idx_start:idx_end]
            if len(segment) == 0:
                continue
            amplitude_mv = np.mean(np.abs(segment)) * 1000
            if emg_threshold[0] <= amplitude_mv <= emg_threshold[1]:
                detected_windows.append((start, end))
                print(f"EMG detected [{format_time(start)} - {format_time(end)}]: type={movement_type}")
        return detected_windows

    def export_csv(head_windows, jaw_windows):
        print(f"Exporting movement events to {output_csv}...")
        all_windows = [(s, e, 'head') for s, e in head_windows] + [(s, e, 'jaw') for s, e in jaw_windows]
        all_windows.sort(key=lambda x: x[0])
        rows = []
        for start, end, mtype in all_windows:
            row = {'timestamp': format_time(start), 'head movement': '', 'jaw movement': '', 'emg detected': ''}
            if mtype == 'head':
                row['head movement'] = 'yes'
            else:
                row['jaw movement'] = 'yes'
            idx_start = int(start * fsBionode)
            idx_end = int(end * fsBionode)
            segment = emg_filtered[idx_start:idx_end]
            amplitude_mv = np.mean(np.abs(segment)) * 1000 if len(segment) > 0 else 0
            if emg_threshold[0] <= amplitude_mv <= emg_threshold[1]:
                row['emg detected'] = 'yes'
            rows.append(row)
        with open(output_csv, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['timestamp', 'head movement', 'jaw movement', 'emg detected'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV export complete: {output_csv}")

    # === Main Execution ===
    emg_raw = load_emg()
    emg_filtered = lowpass_filter(emg_raw)
    head_windows, jaw_windows, fps = extract_movement_windows()
    detect_emg(head_windows, "head")
    detect_emg(jaw_windows, "jaw")
    export_csv(head_windows, jaw_windows)
    print("EMG movement detection complete.")


if __name__ == "__main__":
    BLOCK_PATH = r"/Users/arundhatishankaran/Research/ear3.31.25_1.bin"
    VIDEO_PATH = r"video_recordings/alessandro.mov"
    emgDetect(BLOCK_PATH, VIDEO_PATH)

