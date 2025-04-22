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