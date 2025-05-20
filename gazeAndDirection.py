import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import timedelta
from collections import deque
import time

# Kalman filter for smoothing out any x coordinate noise, was getting a bunch of webcam noise before
class Kalman1D:
    def __init__(self):
        self.x = np.array([[0], [0]])
        self.P = np.eye(2) * 500
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.R = np.array([[10]])
        self.Q = np.array([[1, 0], [0, 3]])

    def update(self, z):
        z = np.array([[z]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0]

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

        # Left eye corners for direction detection later
        self.left_eye_left_corner = 33   # Outer corner (towards temple)
        self.left_eye_right_corner = 133  # Inner corner (towards nose)
        
        # Right eye corners for direction detection
        self.right_eye_left_corner = 362  # Inner corner (towards nose)
        self.right_eye_right_corner = 263  # Outer corner (towards the temple)

        # Iris landmarks for tracking the center
        self.left_iris_indices = [469, 470, 471, 472]
        self.right_iris_indices = [474, 475, 476, 477]

        # Kalman filtering to smooth the iris x-coordinates
        self.kf_left_x = Kalman1D()
        self.kf_right_x = Kalman1D()

        self.prev_left_x = None
        self.prev_right_x = None
        self.eye_velocity_buffer = deque(maxlen=30)
        self.eye_tracking_time_buffer = deque(maxlen=30)
        self.last_eye_tracking_time = time.time()

        # Blink detection parameters
        self.ear_threshold = 0.25
        self.consec_frames = 3
        self.frame_counter = 0
        self.blink_counter = 0
        self.eye_state_history = []
        self.last_recorded_time = -0.05
        
        # Gaze direction thresholds (can change to better calibrate)
        self.left_threshold = 0.40  # If iris is in the left half of the eye, looking left
        self.right_threshold = 0.60  # If iris is in the right half of the eye, looking right
        
        # History for gaze direction tracking, 5 frame rolling window to sort out any wrong direction detection
        self.gaze_direction_history = []
        self.gaze_direction = "CENTER"
        self.gaze_smoothing_frames = 5
        self.gaze_direction_buffer = deque(maxlen=self.gaze_smoothing_frames)

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
        return np.array([(self.landmarks[i].x * w, self.landmarks[i].y * h) for i in indices])

    def _calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio for given eye landmarks"""
        p1, p2, p3, p4, p5, p6 = eye_points
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        ear = (A + B) / (2.0 * C)
        return ear
    
    #this function has the gaze direction detection + the eye tracking logic all in it
    def _detect_gaze_direction(self):
        """Detects if a person is looking left, right, or center"""
        if not self.landmarks:
            return "NO FACE"
            
        h, w = self.frame.shape[:2]
        
        # Gets the eye corner positions
        left_eye_left = (self.landmarks[self.left_eye_left_corner].x * w, 
                         self.landmarks[self.left_eye_left_corner].y * h)
        left_eye_right = (self.landmarks[self.left_eye_right_corner].x * w, 
                          self.landmarks[self.left_eye_right_corner].y * h)
        
        right_eye_left = (self.landmarks[self.right_eye_left_corner].x * w, 
                          self.landmarks[self.right_eye_left_corner].y * h)
        right_eye_right = (self.landmarks[self.right_eye_right_corner].x * w, 
                           self.landmarks[self.right_eye_right_corner].y * h)
        
        # finds where the iris is (this is the iris tracking stuff)
        left_iris_pts = [(self.landmarks[i].x * w, self.landmarks[i].y * h) for i in self.left_iris_indices]
        right_iris_pts = [(self.landmarks[i].x * w, self.landmarks[i].y * h) for i in self.right_iris_indices]
        
        # takes an average of the iris coordinates to determine the center of the iris 
        left_center_x = np.mean([pt[0] for pt in left_iris_pts])
        right_center_x = np.mean([pt[0] for pt in right_iris_pts])
       
        # takes the raw iris coordinates and applies the kalman filter to smooth out the noise
        filtered_left_position = self.kf_left_x.update(left_center_x)
        filtered_right_position = self.kf_right_x.update(right_center_x)
        
        # calculates where the iris is relative to the corners based on a ratio (0 = left corner, 1 = right corner)
        left_eye_width = left_eye_right[0] - left_eye_left[0]
        right_eye_width = right_eye_right[0] - right_eye_left[0]
        
        if left_eye_width == 0 or right_eye_width == 0:  
            return "UNKNOWN"
            
        left_eye_ratio = (filtered_left_position - left_eye_left[0]) / left_eye_width
        right_eye_ratio = (filtered_right_position - right_eye_left[0]) / right_eye_width
        
        # averages the ratios from both eyes for stabilizing
        gaze_ratio = (left_eye_ratio + right_eye_ratio) / 2.0
        
        # determines the direction based on thresholds
        if gaze_ratio < self.left_threshold:
            direction = "RIGHT"
        elif gaze_ratio > self.right_threshold:
            direction = "LEFT"
        else:
            direction = "CENTER"
            
        # adds to the buffer for smoothing 
        self.gaze_direction_buffer.append(direction)
        
        # determines the most common direction in the buffer to sort out any wrong detection
        if len(self.gaze_direction_buffer) == self.gaze_smoothing_frames:
            counts = {}
            for dir in self.gaze_direction_buffer:
                counts[dir] = counts.get(dir, 0) + 1
        
            self.gaze_direction = max(counts, key=counts.get)
            
        return self.gaze_direction, gaze_ratio

    def is_blinking(self, current_time):
        """Detect blink using Eye Aspect Ratio (EAR)"""
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

    def export_to_csv(self, filename="eye_state_log.csv"):
        """Export eye state history to CSV file"""
        df = pd.DataFrame(self.eye_state_history, columns=["Timestamp (s)", "Eye State", "EAR Value"])
        df.to_csv(filename, index=False)
        print(f"Eye state data exported to {filename}")

     # have it as a separate csv rn i can change it to be in the same csv    
    def export_gaze_to_csv(self, filename="gaze_direction_log.csv"):
        """Export gaze direction history to CSV file"""
        df = pd.DataFrame(self.gaze_direction_history, 
                         columns=["Timestamp (s)", "Gaze Direction", "Gaze Ratio"])
        df.to_csv(filename, index=False)
        print(f"Gaze direction data exported to {filename}")

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
            
        # draw eye corner landmarks in a different color
        corner_indices = [self.left_eye_left_corner, self.left_eye_right_corner, 
                         self.right_eye_left_corner, self.right_eye_right_corner]
        for idx in corner_indices:
            pt = self.landmarks[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

        # drawing around the iris 
        left_iris_pts = [(self.landmarks[i].x * w, self.landmarks[i].y * h) for i in self.left_iris_indices]
        right_iris_pts = [(self.landmarks[i].x * w, self.landmarks[i].y * h) for i in self.right_iris_indices]
        left_iris_np = np.array(left_iris_pts, dtype=np.int32)
        right_iris_np = np.array(right_iris_pts, dtype=np.int32)
        cv2.polylines(frame, [left_iris_np], isClosed=True, color=(0, 255, 255), thickness=1)
        cv2.polylines(frame, [right_iris_np], isClosed=True, color=(0, 255, 255), thickness=1)

        # kalman filtering to smooth iris x-center
        left_center_x = np.mean([pt[0] for pt in left_iris_pts])
        right_center_x = np.mean([pt[0] for pt in right_iris_pts])
        self.kf_left_x.predict()
        self.kf_right_x.predict()
        filtered_left_position = self.kf_left_x.update(left_center_x)
        filtered_right_position = self.kf_right_x.update(right_center_x)

        # draws center of iris
        cv2.circle(frame, (int(filtered_left_position), int(np.mean([pt[1] for pt in left_iris_pts]))), 3, (255, 0, 0), -1)
        cv2.circle(frame, (int(filtered_right_position), int(np.mean([pt[1] for pt in right_iris_pts]))), 3, (255, 0, 0), -1)

        # Format timestamp
        timestamp_str = f"{current_time:.2f} s"
        y_pos = 30
        cv2.putText(frame, f"Time: {timestamp_str}", (50, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30

        # Draw eye open/closed state
        left_eye = self._get_eye_points(self.left_eye_indices)
        right_eye = self._get_eye_points(self.right_eye_indices)
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < self.ear_threshold and self.frame_counter >= self.consec_frames:
            cv2.putText(frame, "EYES CLOSED", (50, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "EYES OPEN", (50, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_pos += 40
        
        # Detect and display gaze direction
        gaze_result = self._detect_gaze_direction()
        if isinstance(gaze_result, tuple):
            gaze_direction, gaze_ratio = gaze_result
            # Record gaze data
            if current_time >= self.last_recorded_time + 0.05:
                self.gaze_direction_history.append((current_time, gaze_direction, gaze_ratio))
        else:
            gaze_direction = gaze_result
            gaze_ratio = np.nan
            # Record gaze data
            if current_time >= self.last_recorded_time + 0.05:
                self.gaze_direction_history.append((current_time, gaze_direction, np.nan))
        
        # Display gaze direction with a bigger font and appropriate color
        direction_color = {
            "LEFT": (0, 165, 255),  # Orange
            "RIGHT": (0, 165, 255),  # Orange
            "CENTER": (0, 255, 0),  # Green
            "NO FACE": (0, 0, 255),  # Red
            "UNKNOWN": (128, 128, 128)  # Gray
        }
        
        color = direction_color.get(gaze_direction, (255, 255, 255))
        cv2.putText(frame, f"LOOKING: {gaze_direction}", (50, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Display the ratio for debugging (can be removed in production)
        if not np.isnan(gaze_ratio):
            y_pos += 30
            cv2.putText(frame, f"Gaze ratio: {gaze_ratio:.2f}", (50, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # Mark pose and jaw analysis facial landmarks
        pose_jaw_indices = [33, 263, 1, 61, 291, 199, 17, 0, 14]
        for idx in pose_jaw_indices:
            pt = self.landmarks[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)  # Yellow for pose/jaw landmarks
        return frame

if __name__ == "__main__":
    # Main execution
    cap = cv2.VideoCapture(0)  # put 0 for webcame use 
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
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Export data to CSV when finished
    gaze.export_to_csv()
    gaze.export_gaze_to_csv()