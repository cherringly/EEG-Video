'''
1. EEG Processing
- Loads EEG data from bin file
- Low pass filter (20 Hz cutoff)
- Computes STFT (0.5 sec window, 0.05 sec step)

2. Eye-Tracking (with MediaPipe for eye landmarks)
- Uses EAR (Eye Aspect Ratio) to detect blinks
- If EAR < threshold (0.25), eye is considered closed
- Tracks eye state every 0.05 seconds (20 fps)

3. Synchronization of EEG and Eye-Tracking Data
- Matches EEG spectral power with eye state
- CSV file export
  a. Timestamp (s)
  b. Eye state (OPEN/CLOSED)
  c. EAR value
  d. Power values (0.1-20 Hz)

4. Visualization
- Plots spectrogram with eye-close events overlay
'''

import numpy as np
from scipy import signal
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os
# from bionodebinopen import fn_BionodeBinOpen
import threading
import time
from queue import Queue
import random #EEG simulation data
from matplotlib.animation import FuncAnimation


class RealTimeAnalysis:
    def __init__(self):
        

        # Simulated EEG Configuration
        self.config = {
            'eeg_fs': 250,  # Sampling frequency 
            'high_cutoff': 20,  # Low-pass filter cutoff frequencyqq
            'stft_win_sec': 0.5,  # STFT window length in seconds
            'stft_step_sec': 0.05,  # STFT step size in seconds
            'output_folder': './real_time_results',
            'buffer_size': 10  # Size of data buffer in seconds 
        }

        # Simulated EEG Configuration
        win_samples = int(self.config['stft_win_sec'] * self.config['eeg_fs'])
        step_samples = int(self.config['stft_step_sec'] * self.config['eeg_fs'])
    
        f_init, t_init, Zxx_init = signal.stft(np.zeros(win_samples), 
                        fs=self.config['eeg_fs'],
                        nperseg=win_samples, 
                        noverlap=win_samples - step_samples,
                        nfft=2048
                    )
    
        freq_mask_init = (f_init >= 0.1) & (f_init <= self.config['high_cutoff'])
        self.init_freq_len = sum(freq_mask_init)
        self.init_freqs = f_init[freq_mask_init]


        # Create buffers for real-time data
        self.eeg_buffer = np.array([])
        self.eye_state_history = []
        self.combined_data = []
        # Gaze Tracking Configuration
        self.gaze_tracker = MediaPipeGazeTracking()
        # Ensure output folder exists
        os.makedirs(self.config['output_folder'], exist_ok=True)

        

        # STFT Parameters
        self.stft_params = {
            'win_sec': 0.5,
            'step_sec': 0.05
        }

        # Threading and Queues for Real-time Processing
        self.eeg_queue = Queue(maxsize=1000)
        self.eye_queue = Queue(maxsize=1000)
        self.running = False
        
        # Visualization
        self.fig = None
        self.ax = None
        self.spec_plot = None
        self.ani = None
    
    
    #Initialize Matplotlib figure for real-time plotting
    def init_visualization(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(15, 6)) 
            self.ax.set_title("STFT Spectrogram with Eye Closure Overlay")
            self.ax.set_ylabel('Frequency [Hz]')
            self.ax.set_xlabel('Time [s]')
            plt.tight_layout()

    def start(self, video_path=None):
        """Start real-time processing"""
        self.running = True

        # Initialize visualization
        self.init_visualization()
        # Start EEG and Eye-Tracking threads
        self.eeg_thread = threading.Thread(target=self.simulate_eeg_stream, daemon=True)
        self.eye_thread = threading.Thread(target=self.process_eye_stream, args=(video_path,), daemon=True)
        # start processing thread
        self.processing_thread = threading.Thread(target=self.process_data, daemon=True)

        self.eeg_thread.start()
        self.eye_thread.start()
        self.processing_thread.start()
        print("Starting visualization... Press 'q' in eye tracking window to stop.")

        self.ani = FuncAnimation(
            self.fig, 
            self.update_visualization, 
            interval=100,  # Update every 50 ms
            cache_frame_data=False,
            blit=False
        )

        plt.show(block=True)  


    # Stop processing and export results
    def stop(self):
        self.running = False
        if hasattr(self, 'ani') and self.ani is not None:
            try:
                self.ani.event_source.stop()
            except AttributeError:
                pass
        if hasattr(self, 'eeg_thread'):
            self.eeg_thread.join(timeout=1.0)
        if hasattr(self, 'eye_thread'):
            self.eye_thread.join(timeout=1.0)
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        self.export_results()
        print("Processing stopped and results exported...")

    # Simulate real-time EEG data stream(need to replace with actual data loading)
    def simulate_eeg_stream(self):
        """Simulate real-time EEG data stream"""
        print("starting EEG stream simulation...")
        fs = self.config['eeg_fs']

        
        # Simulate EEG signal with theta, alpha, beta waves + noise
        t = np.linspace(0, 1/fs, fs)
        theta = 10 * np.sin(2 * np.pi * 6 * t)    # θ wave
        alpha = 6 * np.sin(2 * np.pi * 10 * t)   # α wave
        beta = 4 * np.sin(2 * np.pi * 20 * t)    # β wave
        delta = 8 * np.sin(2 * np.pi * 3 * t)    # δ wave
        noise = 0.2 * np.random.randn(fs)        #white Gaussian noise
        
        base_signal = theta + alpha + beta+ delta + noise
        
        try:
            while self.running:
                # Simulate real-time data by sending 10 ms chunks
                chunk = base_signal[:int(fs * 0.01)]
                
                # Randomly simulate blinks (1% chance every 10 ms)
                chunk_size = int(fs * 0.01)
                if len(base_signal) >= chunk_size:
                    chunk = base_signal[:chunk_size]
                    base_signal = base_signal[chunk_size:]

                    if len(base_signal) < chunk_size:
                        t = np.arange(fs) / fs
                        theta = 10 * np.sin(2 * np.pi * 6 * t)
                        alpha = 6 * np.sin(2 * np.pi * 10 * t)
                        beta = 4 * np.sin(2 * np.pi * 20 * t)
                        delta = 8 * np.sin(2 * np.pi * 3 * t)
                        noise = 0.2 * np.random.randn(fs)
                        base_signal = theta + alpha + beta + delta + noise
                    
                    self.eeg_queue.put(chunk)    
                if random.random() < 0.01:
                    blink_duration = 0.1  
                    blink_samples = int(blink_duration * fs)
                    blink_artifact = 50 * np.sin(np.linspace(0, np.pi, blink_samples))
                    self.eeg_queue.put(blink_artifact)
                
                time.sleep(0.01)
        except Exception as e:
            print(f"EEG stream error: {e}")
        finally:
            print("EEG stream simulation ended")
                

    # Process eye-tracking stream
    def process_eye_stream(self, video_path=None):
        print("starting eye-tracking stream...")
        # Open video or webcam
        if video_path:
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)  # Open default webcam
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 30 fps
        frame_interval = 1 / fps
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:  # End of video or webcam error
                    break
                
                current_time = frame_count * frame_interval  # Calculate timestamp
                frame_count += 1
                
                # Process eye-tracking
                self.gaze_tracker.refresh(frame)  # Update landmarks
                self.gaze_tracker.is_blinking(current_time)  # Check for blinks
                
                # Display eye-tracking results
                if self.gaze_tracker.frame is not None:
                    cv2.imshow('Eye Tracking', self.gaze_tracker.frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit(need modify)
                        self.stop()
                        break
                
                # Store eye state data
                self.eye_queue.put((current_time, self.gaze_tracker.last_state, self.gaze_tracker.last_ear))
                time.sleep(max(0, frame_interval - 0.005))  

        except Exception as e:
            print(f"Eye-tracking flow error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Eye-tracking data stream ended")
    
    # Stop processing and export results
    def process_data(self):
        print("Start data processing...")
        fs = self.config['eeg_fs']
        win_samples = int(self.config['stft_win_sec'] * fs)  
        step_samples = int(self.config['stft_step_sec'] * fs)  
        
        # Design low-pass Butterworth filter
        b, a = signal.butter(4, self.config['high_cutoff'] / (fs / 2), btype='low')
        
        try:
            while self.running:
                # 1. Update EEG buffer 
                while not self.eeg_queue.empty():
                    chunk = self.eeg_queue.get()
                    self.eeg_buffer = np.append(self.eeg_buffer, chunk)
                    self.eeg_queue.task_done()
                
                # Limit buffer size to last 10 seconds
                max_buffer_samples = int(self.config['buffer_size'] * fs)
                if len(self.eeg_buffer) > max_buffer_samples:
                    self.eeg_buffer = self.eeg_buffer[-max_buffer_samples:]
                
                # 2. Update eye state history
                while not self.eye_queue.empty():
                    timestamp, state, ear = self.eye_queue.get()
                    self.eye_state_history.append((timestamp, state, ear))
                    # Limit history to last 1000 entries
                    if len(self.eye_state_history) > 1000:
                        self.eye_state_history.pop(0)
                    self.eye_queue.task_done()
                
                # 3. Process EEG data if we have enough samples
                if len(self.eeg_buffer) >= win_samples:
                    # Apply low-pass filter
                    filtered_eeg = signal.filtfilt(b, a, self.eeg_buffer)
                    
                    # Compute STFT
                    f, t, Zxx = signal.stft(
                        filtered_eeg, 
                        fs=fs,
                        nperseg=win_samples,
                        noverlap=win_samples - step_samples,
                        nfft=8192
                    )
                    power = np.abs(Zxx)**2  # Compute power
                    
                    # Filter frequencies
                    freq_mask = (f >= 0.1) & (f <= self.config['high_cutoff'])
                    self.f_filtered = f[freq_mask]
                    self.power_filtered = power[freq_mask, :]

                    # Update STFT times to reflect real-time
                    if hasattr(self, 'init_freq_len'):
                        current_freq_len = len(self.f_filtered)
                    
                        if current_freq_len != self.init_freq_len:
                            if current_freq_len > self.init_freq_len:
                                self.power_filtered = self.power_filtered[:self.init_freq_len, :]
                                self.f_filtered = self.f_filtered[:self.init_freq_len]
                            else:
                            
                                pad_width = ((0, self.init_freq_len - current_freq_len), (0, 0))
                                self.power_filtered = np.pad(self.power_filtered, pad_width, mode='edge')
                    # Adjust STFT times to align with real-time
                    self.stft_times = t + (time.time() - len(self.eeg_buffer)/fs)
                    
                    # Update visualization data
                    self.synchronize_data()
                    
                time.sleep(0.05)  
        except Exception as e:
            print(f"Data processing error: {e}")
        finally:
            print("Data processing completed")


    # Synchronize eye state with EEG spectral data
    def synchronize_data(self):
        if not self.eye_state_history:
            return
            
        # Convert eye state history to DataFrame
        eye_df = pd.DataFrame(
            self.eye_state_history,
            columns=["Timestamp", "Eye_State", "EAR_Value"]
        )
        
        # For each STFT time, find the closest eye state record
        for i, stft_time in enumerate(self.stft_times):
            # Find closest eye state record
            time_diff = np.abs(eye_df["Timestamp"] - stft_time)
            closest_idx = time_diff.idxmin()
            
            # Check if the closest record is within 100 ms
            if time_diff[closest_idx] < 0.1:
                eye_state = eye_df.iloc[closest_idx]["Eye_State"]
                ear_value = eye_df.iloc[closest_idx]["EAR_Value"]
            else:
                eye_state = "no match"
                ear_value = np.nan
            
            # Get frequency power values for this time point
            freq_powers = {
                f"{freq:.1f} Hz": self.power_filtered[j, i] 
                for j, freq in enumerate(self.f_filtered)
            }
            
            self.combined_data.append({
                "Timestamp": stft_time,
                "Eye_State": eye_state,
                "EAR_Value": ear_value,
                **freq_powers
            })
            
            # Limit combined data size to last 5000 entries
            if len(self.combined_data) > 10000:
                self.combined_data = self.combined_data[-5000:]

    # Update visualization with latest STFT and eye state
    def update_visualization(self, frame):
        try:
            self.ax.clear()
            self.ax.set_title("STFT Spectrogram with Eye Closure Overlay")
            self.ax.set_ylabel('Frequency [Hz]')
            self.ax.set_xlabel('Time [s]')
            # Plot spectrogram
            if hasattr(self, 'power_filtered') and hasattr(self, 'stft_times'):
                if self.power_filtered.size > 0 and len(self.stft_times) > 0:
                    display_length = min(20, self.power_filtered.shape[1])

                    if (display_length > 0 and 
                        self.power_filtered.shape[0] == self.init_freq_len and
                        self.power_filtered.shape[1] >= display_length):
                        # Plot the latest STFT data
                        latest_data = self.power_filtered[:, -display_length:]
                        if latest_data.shape == (self.init_freq_len, display_length):
                            spec_data_clipped = np.clip(latest_data, 1e-10, np.inf)
                            log_spec = 10 * np.log10(spec_data_clipped)
                            # Plot spectrogram
                            if len(self.stft_times) >= display_length:
                                start_time = self.stft_times[-display_length]
                                end_time = self.stft_times[-1]
                                t_display = np.linspace(start_time, end_time, display_length)
                                T, F = np.meshgrid(t_display, self.init_freqs)

                                spec_plot = self.ax.pcolormesh(
                                T, F, log_spec, 
                                shading='gouraud', 
                                cmap='jet'
                            )
                                self.fig.colorbar(spec_plot, ax=self.ax, label='Power [dB]')
            # Overlay eye closure events
            for timestamp, state, _ in self.eye_state_history:
                if state == "CLOSED":
                    self.ax.axvspan(
                    timestamp - 0.025,  
                    timestamp + 0.025,   
                    color='blue',       
                    alpha=0.01,          
                    zorder=5  
                )
        except Exception as e:
            print(f"Visualization update error: {e}")                    


    # Export combined results to CSV
    def export_results(self):
        if not self.combined_data:
            print("no data to export")
            return    
        output_path = os.path.join(
            self.config['output_folder'], 
            f"real_time_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df = pd.DataFrame(self.combined_data)
        df.to_csv(output_path, index=False)
        print(f"Results exported to: {output_path}")
        return df
    





class MediaPipeGazeTracking:
    # Eye Aspect Ratio (EAR) based gaze tracking using MediaPipe
    def __init__(self):
        self.frame = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]
        self.ear_threshold = 0.25
        self.consec_frames = 3
        self.frame_counter = 0
        self.blink_counter = 0
        self.eye_state_history = []
        self.last_recorded_time = -0.05
        self.last_state = None 
        self.last_ear = None

        # MediaPipe drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0)
        )


    # Refresh the frame and process landmarks
    def refresh(self, frame):
        # self.frame = frame
        # self.landmarks = None
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # results = self.face_mesh.process(rgb)
        # if results.multi_face_landmarks:
        #     self.landmarks = results.multi_face_landmarks[0].landmark
        self.frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        self.landmarks = None
        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0].landmark
            # Draw landmarks for visualization
            self.mp_drawing.draw_landmarks(
                image=self.frame,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec
            )


    # Get eye points from landmarks
    def _get_eye_points(self, indices):
        # h, w = self.frame.shape[:2]
        # return np.array([(self.landmarks[i].x * w, self.landmarks[i].y * h) 
        #                 for i in indices])
        if not self.landmarks or self.frame is None:
            return None
        h, w = self.frame.shape[:2]
        return np.array([
            (self.landmarks[i].x * w, self.landmarks[i].y * h) 
            for i in indices
        ])


    # Calculate Eye Aspect Ratio (EAR)
    def _calculate_ear(self, eye_points):
        p1, p2, p3, p4, p5, p6 = eye_points
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        return (A + B) / (2.0 * C) if C != 0 else 0.0


    # Check if the eye is blinking
    def is_blinking(self, current_time):
        if not self.landmarks:
            self.last_state = "NO FACE"
            self.last_ear = np.nan
            if current_time >= self.last_recorded_time + 0.05:
                self.eye_state_history.append((current_time, "NO FACE", np.nan))
                self.last_recorded_time = current_time
            return False

        left_eye = self._get_eye_points(self.left_eye_indices)
        right_eye = self._get_eye_points(self.right_eye_indices)
        if left_eye is None or right_eye is None:
            self.last_state = "NO FACE"
            self.last_ear = np.nan
            return False
        
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        self.last_ear = ear
        
        if ear < self.ear_threshold:
            eye_state = "CLOSED"
            self.frame_counter += 1
        else:
            eye_state = "OPEN"
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
            self.frame_counter = 0
        
        if current_time >= self.last_recorded_time + 0.05:
            self.eye_state_history.append((current_time, eye_state, ear))
            self.last_recorded_time = current_time
        
        return eye_state == "CLOSED" and self.frame_counter >= self.consec_frames




# Main execution
if __name__ == "__main__":
    analyzer = None
    try:
        analyzer = RealTimeAnalysis()
        analyzer = RealTimeAnalysis()
        # Start real-time processing
        analyzer.start()  # Replace with video path if needed
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping...")
    except Exception as e:
        print(f"error: {e}")
    finally:
        if 'analyzer' in locals():
            analyzer.stop()