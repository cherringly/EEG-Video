# head_jaw_tracker.py

import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe drawing and face mesh solutions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

class HeadJawTracker:
    def __init__(self):
        # Initialize face mesh with landmark refinement
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

        # Calibration settings
        self.calibration_frames = 60  # Number of frames to use for calibration
        self.frame_count = 0  # Counter for frames processed during calibration

        # Lists to store values for calculating calibration offsets
        self.x_values = []  # Collect pitch values during calibration
        self.y_values = []  # Collect yaw values during calibration
        self.z_values = []  # Collect roll values during calibration
        self.jaw_values = []  # Collect jaw deltas (lip distance) during calibration

        self.calibrated = False  # Flag to indicate calibration is complete

        # Offset values to normalize head pose based on calibration
        self.x_offset = 0  # Mean pitch offset
        self.y_offset = 0  # Mean yaw offset
        self.z_offset = 0  # Mean roll offset

        # Jaw state determined from calibration average
        self.jaw_neutral = "Neutral"
        self.jaw_neutral_value = 0  # Average value of jaw distance after calibration

    def get_landmark_points(self, face_landmarks, img_w, img_h):
        # List of important facial landmark indexes
        idxs = [33, 263, 1, 61, 291, 199, 17, 0, 14]  # Eyes, nose tip, lips, chin

        face_3d = []  # 3D coordinates used for pose estimation
        face_2d = []  # 2D coordinates projected on image

        nose_2d = (0, 0)  # Used for projecting nose direction vector
        nose_3d = (0, 0, 0)

        jaw_y = top_y = bottom_y = 0  # Y-coordinates to estimate jaw status

        for idx, lm in enumerate(face_landmarks.landmark):
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            if idx in idxs:
                if idx == 1:
                    # Nose tip
                    nose_2d = (x, y)
                    nose_3d = (x, y, lm.z * 3000)  # Z is scaled for visibility
                if idx == 0:
                    # Top lip position
                    top_y = y
                if idx == 14:
                    # Bottom lip position
                    bottom_y = y
                if idx == 17:
                    # Chin position
                    jaw_y = y

                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

        return np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64), nose_2d, nose_3d, jaw_y, top_y, bottom_y

    def get_head_rotation(self, face_2d, face_3d, img_w, img_h):
        # Define camera intrinsic matrix based on image dimensions
        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_w / 2],
                               [0, focal_length, img_h / 2],
                               [0, 0, 1]])

        # Assume no lens distortion
        dist_matrix = np.zeros((4, 1))

        # Solve PnP to get rotation vector and translation vector
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Convert rotation vector to matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Decompose rotation matrix to get pitch (x), yaw (y), roll (z)
        angles, *_ = cv2.RQDecomp3x3(rmat)

        # Convert radians to degrees
        x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
        return x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix

    def get_jaw_state(self, top_y, bottom_y):
        # Estimate jaw state based on difference from calibrated neutral value
        diff = bottom_y - top_y
        deviation = diff - self.jaw_neutral_value
        if deviation > 3:
            return "Jaw Dropped"
        elif deviation < -1.5:
            return "Jaw Clenched"
        return "Neutral"

    def draw_feedback(self, image, rotation, jaw_state):
        # Draw pitch, yaw, roll, and jaw status on screen overlay
        x, y, z = rotation
        cv2.putText(image, f"Pitch: {x:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"Yaw:   {y:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"Roll:  {z:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"Jaw:   {jaw_state}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    face_2d, face_3d, nose_2d, nose_3d, jaw_y, top_y, bottom_y = self.get_landmark_points(face_landmarks, w, h)
                    x, y, z, rot_vec, trans_vec, cam_matrix, dist_matrix = self.get_head_rotation(face_2d, face_3d, w, h)
                    jaw_diff = bottom_y - top_y

                    # Calibration logic to compute average baseline offsets
                    if not self.calibrated:
                        self.x_values.append(x)
                        self.y_values.append(y)
                        self.z_values.append(z)
                        self.jaw_values.append(jaw_diff)
                        self.frame_count += 1

                        # Once enough frames are gathered, compute average offsets
                        if self.frame_count == self.calibration_frames:
                            self.x_offset = np.mean(self.x_values)
                            self.y_offset = np.mean(self.y_values)
                            self.z_offset = np.mean(self.z_values)
                            self.jaw_neutral_value = np.mean(self.jaw_values)
                            self.calibrated = True

                    if not self.calibrated:
                        x = y = z = 0
                        jaw_state = "Neutral"
                    else:
                        # Apply calibration offsets to normalize values
                        x -= self.x_offset
                        y -= self.y_offset
                        z -= self.z_offset
                        jaw_state = self.get_jaw_state(top_y, bottom_y)

                    self.draw_feedback(frame, (x, y, z), jaw_state)

            # Show the processed video frame
            cv2.imshow("Head & Jaw Tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Launch tracker
    tracker = HeadJawTracker()
    tracker.run()
