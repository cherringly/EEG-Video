import cv2
import mediapipe as mp
import numpy as np

class MediaPipeGazeTracking:
    def __init__(self):
        self.frame = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                         max_num_faces=1,
                                                         refine_landmarks=True,
                                                         min_detection_confidence=0.5,
                                                         min_tracking_confidence=0.5)
        self.blink_threshold = 0.23  # tweakable
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]

    def refresh(self, frame):
        self.frame = frame
        self.landmarks = None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0].landmark

    def _eye_aspect_ratio(self, eye_points):
        p1 = np.array([eye_points[1].x, eye_points[1].y])
        p2 = np.array([eye_points[5].x, eye_points[5].y])
        p3 = np.array([eye_points[2].x, eye_points[2].y])
        p4 = np.array([eye_points[4].x, eye_points[4].y])
        p0 = np.array([eye_points[0].x, eye_points[0].y])
        p5 = np.array([eye_points[3].x, eye_points[3].y])

        vertical1 = np.linalg.norm(p2 - p4)
        vertical2 = np.linalg.norm(p3 - p5)
        horizontal = np.linalg.norm(p0 - p1)

        return (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)

    def is_blinking(self):
        if not self.landmarks:
            return False

        left_eye = [self.landmarks[i] for i in self.left_eye_indices]
        right_eye = [self.landmarks[i] for i in self.right_eye_indices]
        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        ear_avg = (left_ear + right_ear) / 2.0

        return ear_avg < self.blink_threshold

    def annotated_frame(self):
        if not self.landmarks:
            return self.frame

        frame = self.frame.copy()
        h, w, _ = frame.shape
        for idx in self.left_eye_indices + self.right_eye_indices:
            pt = self.landmarks[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        return frame



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
