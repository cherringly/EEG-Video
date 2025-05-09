'''

'''

import numpy as np
from scipy import signal
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os
from bionodebinopen import fn_BionodeBinOpen


class CombinedAnalysis:
    def __init__(self):
        # EEG Configuration
        self.eeg_config = {
            'blockPath': '/Users/maryzhang/Downloads/EarEEG_Matlab/Data/25-31-03/2025.03.31.17.53.13_AAO-18.bin',
            'outputFolder': '/Users/maryzhang/Downloads/EarEEG_Matlab',
            'channel': 1,
            'fsBionode': (25e3)/2,
            'ADCres': 12,
            'highCutoff': 20
        }
        
        # Gaze Tracking Configuration
        self.gaze_tracker = MediaPipeGazeTracking()
        
        # STFT Parameters
        self.stft_params = {
            'win_sec': 0.5,
            'step_sec': 0.05
        }
        
        # Combined Data Storage
        self.combined_data = []
        
    def load_eeg_data(self):
        """Load and preprocess EEG data"""
        print("Loading EEG data...")
        earData = fn_BionodeBinOpen(self.eeg_config['blockPath'], 
                                   self.eeg_config['ADCres'], 
                                   self.eeg_config['fsBionode'])
        
        rawChaB = np.array(earData['channelsData'])
        rawChaB = (rawChaB - 2**11) * 1.8 / (2**12 * 1000)
        
        # Apply low-pass filter
        bB, aB = signal.butter(4, self.eeg_config['highCutoff'] / (self.eeg_config['fsBionode'] / 2), 
                  btype='low')
        self.PPfiltChaB = signal.filtfilt(bB, aB, rawChaB[self.eeg_config['channel']-1, :])
        
    def process_video(self, video_path):
        """Process video for eye state detection"""
        print("Processing video for eye state detection...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            frame_count += 1
            
            self.gaze_tracker.refresh(frame)
            self.gaze_tracker.is_blinking(current_time)
            
        cap.release()
        cv2.destroyAllWindows()
        
    def compute_stft(self):
        """Compute STFT on EEG data"""
        print("Computing STFT...")
        fs = self.eeg_config['fsBionode']
        win_samples = int(self.stft_params['win_sec'] * fs)
        step_samples = int(self.stft_params['step_sec'] * fs)
        
        f, t, Zxx = signal.stft(self.PPfiltChaB, fs=fs,
                               nperseg=win_samples,
                               noverlap=win_samples-step_samples)
        self.power = np.abs(Zxx)**2
        
        # Filter frequencies
        freq_mask = (f >= 0.1) & (f <= self.eeg_config['highCutoff'])
        self.f_filtered = f[freq_mask]
        self.power_filtered = self.power[freq_mask, :]
        self.stft_times = t
        
    def synchronize_data(self):
        """Combine eye state and EEG spectral data"""
        print("Synchronizing data...")
        
        # Get eye state data
        eye_data = pd.DataFrame(self.gaze_tracker.eye_state_history,
                              columns=["Timestamp (s)", "Eye State", "EAR Value"])
        
        # Get STFT data
        stft_df = pd.DataFrame(self.power_filtered.T,
                             index=self.stft_times,
                             columns=[f"{freq:.1f} Hz" for freq in self.f_filtered])
        
        # Merge dataframes using nearest timestamp matching
        for stft_time in stft_df.index:
            # Find closest eye state measurement
            time_diff = np.abs(eye_data["Timestamp (s)"] - stft_time)
            closest_idx = time_diff.idxmin()
            
            if time_diff[closest_idx] < 0.1:  # Only match if within 100ms
                eye_state = eye_data.iloc[closest_idx]["Eye State"]
                ear_value = eye_data.iloc[closest_idx]["EAR Value"]
            else:
                eye_state = "NO MATCH"
                ear_value = np.nan
                
            # Get frequency power values
            freq_powers = stft_df.loc[stft_time].to_dict()
            
            # Store combined data
            self.combined_data.append({
                "Timestamp": stft_time,
                "Eye_State": eye_state,
                "EAR_Value": ear_value,
                **freq_powers
            })
            
    def export_results(self):
        """Export combined results to CSV"""
        os.makedirs(self.eeg_config['outputFolder'], exist_ok=True)
        output_path = os.path.join(self.eeg_config['outputFolder'], 'combined_analysis_1.csv')
        
        df = pd.DataFrame(self.combined_data)
        df.to_csv(output_path, index=False)
        print(f"Combined analysis exported to {output_path}")
        
        return df
    
    def plot_spectrogram_with_eye_state(self):
        """Plot STFT Spectrogram with Eye State Overlay"""

        print("Plotting spectrogram with eye state overlay...")

        t = self.stft_times
        f = self.f_filtered
        Sxx = self.power_filtered

        plt.figure(figsize=(15, 6))
        plt.title("STFT Spectrogram with Eye Closure Overlay")

        # Plot spectrogram
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='jet')
        plt.colorbar(label='Power [dB]')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')

        # Overlay vertical regions where eyes are closed
        for timestamp, state, _ in self.gaze_tracker.eye_state_history:
            if state == "CLOSED":
                plt.axvspan(timestamp - 0.025, timestamp + 0.025, color='blue', alpha=0.01)

        plt.tight_layout()
        plt.show()


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
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]
        self.ear_threshold = 0.25
        self.consec_frames = 3
        self.frame_counter = 0
        self.blink_counter = 0
        self.eye_state_history = []
        self.last_recorded_time = -0.05

    def refresh(self, frame):
        self.frame = frame
        self.landmarks = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0].landmark

    def _get_eye_points(self, indices):
        h, w = self.frame.shape[:2]
        return np.array([(self.landmarks[i].x * w, self.landmarks[i].y * h) 
                        for i in indices])

    def _calculate_ear(self, eye_points):
        p1, p2, p3, p4, p5, p6 = eye_points
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        return (A + B) / (2.0 * C)

    def is_blinking(self, current_time):
        if not self.landmarks:
            if current_time >= self.last_recorded_time + 0.05:
                self.eye_state_history.append((current_time, "NO FACE", np.nan))
                self.last_recorded_time = current_time
            return False

        left_eye = self._get_eye_points(self.left_eye_indices)
        right_eye = self._get_eye_points(self.right_eye_indices)
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
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
    analyzer = CombinedAnalysis()
    
    # Process EEG data
    analyzer.load_eeg_data()
    analyzer.compute_stft()
    
    # Process video data (replace with your video path)
    analyzer.process_video("video_recordings/alessandro.mov")
    analyzer.plot_spectrogram_with_eye_state()

    
    # Combine and export results
    analyzer.synchronize_data()
    results_df = analyzer.export_results()
    
    print("Analysis complete!")