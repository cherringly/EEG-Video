import cv2
import mediapipe as mp
import numpy as np

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
        self.baseline_distance = None
        self.blink_threshold_ratio = 0.7  # Eyes must come at least 30% closer to count as blink

    def refresh(self, frame):
        self.frame = frame
        self.landmarks = None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0].landmark
            # Update baseline distance if not set
            if self.baseline_distance is None:
                self._update_baseline_distance()

    def _update_baseline_distance(self):
        """Calculate normal distance between eye centers for reference"""
        left_eye = self._get_eye_points(self.left_eye_indices)
        right_eye = self._get_eye_points(self.right_eye_indices)
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        self.baseline_distance = np.linalg.norm(left_center - right_center)

    def _get_eye_points(self, indices):
        """Convert landmarks to numpy array of eye points"""
        h, w = self.frame.shape[:2]
        return np.array([(self.landmarks[i].x * w, self.landmarks[i].y * h) 
                        for i in indices])

    def is_blinking(self):
        """Detect blink based on eyes moving closer together"""
        if not self.landmarks or self.baseline_distance is None:
            return False

        left_eye = self._get_eye_points(self.left_eye_indices)
        right_eye = self._get_eye_points(self.right_eye_indices)
        
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        current_distance = np.linalg.norm(left_center - right_center)
        
        # Blink detected if eyes are significantly closer than baseline
        return current_distance < self.baseline_distance * self.blink_threshold_ratio

    def annotated_frame(self):
        if not self.landmarks:
            return self.frame

        frame = self.frame.copy()
        h, w, _ = frame.shape
        
        # Draw eye landmarks
        for idx in self.left_eye_indices + self.right_eye_indices:
            pt = self.landmarks[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Add distance info if baseline is set
        if self.baseline_distance is not None:
            left_eye = self._get_eye_points(self.left_eye_indices)
            right_eye = self._get_eye_points(self.right_eye_indices)
            left_center = np.mean(left_eye, axis=0).astype(int)
            right_center = np.mean(right_eye, axis=0).astype(int)
            
            # Draw line between eyes
            cv2.line(frame, tuple(left_center), tuple(right_center), (255, 0, 0), 2)
            
            # Display distance ratio
            current_distance = np.linalg.norm(left_center - right_center)
            ratio = current_distance / self.baseline_distance
            cv2.putText(frame, f"Dist Ratio: {ratio:.2f}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame


# Main execution
cap = cv2.VideoCapture("3.31.25_1.mov")
gaze = MediaPipeGazeTracking()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    if gaze.is_blinking():
        cv2.putText(frame, "BLINKING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Gaze Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()