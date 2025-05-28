import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


class HeadJawTracker:
    def __init__(self):
        # initialize face mesh detector with landmarks refinement
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        # number of frames to use for calibration
        self.calibration_frames = 60
        self.frame_count = 0
        # lists to store rotation and jaw values during calibration
        self.x_values, self.y_values, self.z_values, self.jaw_values = [], [], [], []
        self.calibrated = False
        # calibration offsets for pitch (x), yaw (y), roll (z)
        self.x_offset = self.y_offset = self.z_offset = 0
        # reference jaw distance from top lip to bottom lip
        self.jaw_neutral_value = 0
        
        

    def get_landmark_points(self, face_landmarks, img_w, img_h):
        # returns selected 2d and 3d facial landmarks and y-coordinates for jaw analysis
        idxs = [33, 263, 1, 61, 291, 199, 17, 0, 14]  # key landmarks for pose and jaw
        face_3d, face_2d = [], []
        nose_2d = nose_3d = (0, 0, 0)
        jaw_y = top_y = bottom_y = 0

        for idx, lm in enumerate(face_landmarks.landmark):
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            if idx in idxs:
                if idx == 1: nose_2d, nose_3d = (x, y), (x, y, lm.z * 3000)
                if idx == 0: top_y = y
                if idx == 14: bottom_y = y
                if idx == 17: jaw_y = y
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
        return np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64), top_y, bottom_y




    def get_head_rotation(self, face_2d, face_3d, img_w, img_h):
        # estimate head rotation angles using solvePnP
        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_w / 2],
                               [0, focal_length, img_h / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1))
        _, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
        return x, y, z




    def get_jaw_state(self, top_y, bottom_y):
        # determine jaw state based on deviation from neutral distance
        diff = bottom_y - top_y
        deviation = diff - self.jaw_neutral_value
        if deviation > 3:
            return "Jaw Dropped"
        elif deviation < -1.5:
            return "Jaw Clenched"
        return "Neutral"




    def process(self, frame, face_landmarks):
        # main method to calculate head angles and jaw state
        h, w, _ = frame.shape
        face_2d, face_3d, top_y, bottom_y = self.get_landmark_points(face_landmarks, w, h)
        x, y, z = self.get_head_rotation(face_2d, face_3d, w, h)
        jaw_diff = bottom_y - top_y

        if not self.calibrated:
            # collect data for calibration phase
            self.x_values.append(x)
            self.y_values.append(y)
            self.z_values.append(z)
            self.jaw_values.append(jaw_diff)
            self.frame_count += 1
            if self.frame_count == self.calibration_frames:
                # compute average offsets for neutral position
                self.x_offset = np.mean(self.x_values)
                self.y_offset = np.mean(self.y_values)
                self.z_offset = np.mean(self.z_values)
                self.jaw_neutral_value = np.mean(self.jaw_values)
                self.calibrated = True
            return frame, 0, 0, 0, "Neutral"

        # subtract baseline offsets after calibration
        x -= self.x_offset
        y -= self.y_offset
        z -= self.z_offset
        jaw_state = self.get_jaw_state(top_y, bottom_y)

        # draw pitch, yaw, roll, jaw state on the frame
        cv2.putText(frame, f"Pitch: {x:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Yaw:   {y:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Roll:  {z:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Jaw:   {jaw_state}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return frame, x, y, z, jaw_state








class MediaPipeGazeTracking:
    def __init__(self):
        # initialize face mesh and define eye landmark indices
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]
        self.ear_threshold = 0.25  # eye aspect ratio threshold
        self.consec_frames = 3  # frames to confirm blink
        self.frame_counter = 0
        self.blink_counter = 0
        self.eye_state_history = []  # store (time, eye state, ear)
        self.last_recorded_time = -0.05


    def _get_eye_points(self, landmarks, indices, w, h):
        # extract and return eye landmarks as 2d points
        return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])


    def _calculate_ear(self, eye):
        # calculate eye aspect ratio (EAR) from 6 eye landmarks
        p1, p2, p3, p4, p5, p6 = eye
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        return (A + B) / (2.0 * C)


    def analyze(self, frame, landmarks, current_time):
        # compute eye state and log data if needed
        h, w, _ = frame.shape
        left_eye = self._get_eye_points(landmarks, self.left_eye_indices, w, h)
        right_eye = self._get_eye_points(landmarks, self.right_eye_indices, w, h)
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

        # store state every 0.05s
        if current_time >= self.last_recorded_time + 0.05:
            self.eye_state_history.append((current_time, eye_state, ear))
            self.last_recorded_time = current_time

        y_pos = 150
        if ear < self.ear_threshold and self.frame_counter >= self.consec_frames:
            cv2.putText(frame, "EYES CLOSED", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "EYES OPEN", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame



    def export_to_csv(self, filename="eye_state_log.csv"):
        # save recorded eye states to a csv file
        df = pd.DataFrame(self.eye_state_history, columns=["Timestamp (s)", "Eye State", "EAR Value"])
        df.to_csv(filename, index=False)
        print(f"Eye state data exported to {filename}")






if __name__ == "__main__":
    # open video file
    cap = cv2.VideoCapture("video_recordings/5.24_tdt_e.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    head_tracker = HeadJawTracker()
    gaze_tracker = MediaPipeGazeTracking()

    # main loop for processing video
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
                frame, pitch, yaw, roll, jaw_state = head_tracker.process(frame, landmarks)
                frame = gaze_tracker.analyze(frame, landmarks.landmark, current_time)
                gaze_tracker.export_to_csv()
            else:
                cv2.putText(frame, "NO FACE DETECTED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Combined Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    import matplotlib.pyplot as plt

    # plot timeline of open/closed eyes
    timestamps = [t for (t, state, ear) in gaze_tracker.eye_state_history]
    states_numeric = [1 if state == "OPEN" else 0 for (_, state, _) in gaze_tracker.eye_state_history]

    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, states_numeric, drawstyle="steps-post", color='green', label='Eyes Open')
    plt.yticks([0, 1], ['Closed', 'Open'])
    plt.xlabel("Time (s)")
    plt.ylabel("Eye State")
    plt.title("Eye State Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
