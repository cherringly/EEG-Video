import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque
import time

# Kalman filter for smoothing 1D signals
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

        # Eye & iris landmarks
        self.left_eye_indices   = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices  = [362, 385, 387, 263, 373, 380]
        self.left_eye_left_corner   = 33
        self.left_eye_right_corner  = 133
        self.right_eye_left_corner  = 362
        self.right_eye_right_corner = 263
        self.left_iris_indices  = [469, 470, 471, 472]
        self.right_iris_indices = [474, 475, 476, 477]

        # Kalman filters for x & y
        self.kf_left_x   = Kalman1D()
        self.kf_right_x  = Kalman1D()
        self.kf_left_y   = Kalman1D()
        self.kf_right_y  = Kalman1D()

        # Previous filtered centers
        self.prev_left_x   = None
        self.prev_right_x  = None
        self.prev_left_y   = None
        self.prev_right_y  = None

        # Movement thresholds
        self.movement_threshold = 5.0    # horizontal pixels
        self.vert_threshold     = 5.0    # vertical pixels
        self.movement_history   = []

        # Blink detection
        self.ear_threshold     = 0.25
        self.consec_frames     = 3
        self.frame_counter     = 0
        self.blink_counter     = 0
        self.eye_state_history = []
        self.last_recorded_time= -0.05

        # Gaze direction
        self.left_threshold   = 0.35
        self.right_threshold  = 0.65
        self.gaze_buffer_len  = 5
        self.gaze_direction_buffer = deque(maxlen=self.gaze_buffer_len)
        self.gaze_direction   = "CENTER"
        self.gaze_history     = []

    def refresh(self, frame):
        self.frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        self.landmarks = results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None

    def _get_eye_points(self, indices):
        h, w = self.frame.shape[:2]
        return np.array([(self.landmarks[i].x * w, self.landmarks[i].y * h) for i in indices])

    def _calculate_ear(self, points):
        p1, p2, p3, p4, p5, p6 = points
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        return (A + B) / (2.0 * C)

    def is_blinking(self, t):
        if not self.landmarks:
            if t >= self.last_recorded_time + 0.05:
                self.eye_state_history.append((t, "NO FACE", np.nan))
                self.last_recorded_time = t
            return False

        left_pts  = self._get_eye_points(self.left_eye_indices)
        right_pts = self._get_eye_points(self.right_eye_indices)
        ear = (self._calculate_ear(left_pts) + self._calculate_ear(right_pts)) / 2.0

        if ear < self.ear_threshold:
            self.frame_counter += 1
            state = "CLOSED"
        else:
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
            self.frame_counter = 0
            state = "OPEN"

        if t >= self.last_recorded_time + 0.05:
            self.eye_state_history.append((t, state, ear))
            self.last_recorded_time = t

        return state == "CLOSED" and self.frame_counter >= self.consec_frames

    def _detect_gaze_direction(self):
        if not self.landmarks:
            return "NO FACE"
        h, w = self.frame.shape[:2]

        # eye corners
        le_l = self.landmarks[self.left_eye_left_corner]
        le_r = self.landmarks[self.left_eye_right_corner]
        re_l = self.landmarks[self.right_eye_left_corner]
        re_r = self.landmarks[self.right_eye_right_corner]
        le_l = (le_l.x*w, le_l.y*h)
        le_r = (le_r.x*w, le_r.y*h)
        re_l = (re_l.x*w, re_l.y*h)
        re_r = (re_r.x*w, re_r.y*h)

        # iris raw
        left_pts  = [(self.landmarks[i].x*w, self.landmarks[i].y*h) for i in self.left_iris_indices]
        right_pts = [(self.landmarks[i].x*w, self.landmarks[i].y*h) for i in self.right_iris_indices]
        left_cx  = np.mean([p[0] for p in left_pts])
        right_cx = np.mean([p[0] for p in right_pts])

        # Kalman-filter x
        filtered_lx = self.kf_left_x.update(left_cx)
        filtered_rx = self.kf_right_x.update(right_cx)

        # compute ratio
        lw = le_r[0] - le_l[0]
        rw = re_r[0] - re_l[0]
        if lw == 0 or rw == 0:
            return "UNKNOWN"
        lr = (filtered_lx - le_l[0]) / lw
        rr = (filtered_rx - re_l[0]) / rw
        ratio = (lr + rr) / 2.0

        if ratio < self.left_threshold:   dir_ = "RIGHT"
        elif ratio > self.right_threshold: dir_ = "LEFT"
        else:                              dir_ = "CENTER"

        self.gaze_direction_buffer.append(dir_)
        if len(self.gaze_direction_buffer) == self.gaze_buffer_len:
            counts = {d: self.gaze_direction_buffer.count(d) for d in set(self.gaze_direction_buffer)}
            self.gaze_direction = max(counts, key=counts.get)

        return self.gaze_direction, ratio

    def export_to_csv(self, fn="eye_state_log.csv"):
        pd.DataFrame(self.eye_state_history,
                     columns=["Timestamp","Eye State","EAR"]).to_csv(fn, index=False)
        print(f"Saved eye state log to {fn}")

    def export_gaze_to_csv(self, fn="gaze_log.csv"):
        pd.DataFrame(self.gaze_history,
                     columns=["Timestamp","Gaze Dir","Ratio"]).to_csv(fn, index=False)
        print(f"Saved gaze log to {fn}")

    def export_movement_to_csv(self, fn="movement_log.csv"):
        pd.DataFrame(self.movement_history,
                     columns=["Timestamp","Direction","Delta"]).to_csv(fn, index=False)
        print(f"Saved movement log to {fn}")

    def annotated_frame(self, t):
        blink = self.is_blinking(t)
        frame = self.frame.copy()
        h, w, _ = frame.shape

        if not self.landmarks:
            cv2.putText(frame, "NO FACE", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            return frame

        # draw eye landmarks
        for idx in self.left_eye_indices + self.right_eye_indices:
            p = self.landmarks[idx]
            cv2.circle(frame, (int(p.x*w), int(p.y*h)), 2, (0,255,0), -1)

        corners = [self.left_eye_left_corner, self.left_eye_right_corner,
                   self.right_eye_left_corner, self.right_eye_right_corner]
        for idx in corners:
            p = self.landmarks[idx]
            cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, (255,0,255), -1)

        left_pts  = [(self.landmarks[i].x*w, self.landmarks[i].y*h) for i in self.left_iris_indices]
        right_pts = [(self.landmarks[i].x*w, self.landmarks[i].y*h) for i in self.right_iris_indices]
        cv2.polylines(frame, [np.array(left_pts, np.int32)],  True, (0,255,255),1)
        cv2.polylines(frame, [np.array(right_pts, np.int32)], True, (0,255,255),1)

        # raw centers
        left_cx  = np.mean([p[0] for p in left_pts])
        right_cx = np.mean([p[0] for p in right_pts])
        left_cy  = np.mean([p[1] for p in left_pts])
        right_cy = np.mean([p[1] for p in right_pts])

        # predict & update Kalman
        self.kf_left_x.predict();  self.kf_right_x.predict()
        self.kf_left_y.predict();  self.kf_right_y.predict()
        filt_lx = self.kf_left_x.update(left_cx)
        filt_rx = self.kf_right_x.update(right_cx)
        filt_ly = self.kf_left_y.update(left_cy)
        filt_ry = self.kf_right_y.update(right_cy)

        # draw filtered centers
        cv2.circle(frame, (int(filt_lx), int(filt_ly)), 3, (255,0,0), -1)
        cv2.circle(frame, (int(filt_rx), int(filt_ry)), 3, (255,0,0), -1)

        # movement detection
        h_move = False; v_move = False
        h_dir  = None; v_dir = None

        # horizontal
        if not blink and self.prev_left_x is not None:
            dx = ((filt_lx - self.prev_left_x) + (filt_rx - self.prev_right_x)) / 2.0
            if abs(dx) > self.movement_threshold:
                h_move = True
                h_dir = "RIGHT" if dx > 0 else "LEFT"
                self.movement_history.append((t, h_dir, float(dx)))
        else:
            self.prev_left_x = self.prev_right_x = None

        # vertical
        if not blink and self.prev_left_y is not None:
            dy = ((filt_ly - self.prev_left_y) + (filt_ry - self.prev_right_y)) / 2.0
            if abs(dy) > self.vert_threshold:
                v_move = True
                v_dir = "DOWN" if dy > 0 else "UP"
                self.movement_history.append((t, v_dir, float(dy)))
        else:
            self.prev_left_y = self.prev_right_y = None

        # update prevs
        if not blink:
            self.prev_left_x  = filt_lx
            self.prev_right_x = filt_rx
            self.prev_left_y  = filt_ly
            self.prev_right_y = filt_ry

        # overlay info
        y = 30
        cv2.putText(frame, f"Time: {t:.2f}s", (50,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        y += 30

        # eyes open/closed
        left_eye_pts  = self._get_eye_points(self.left_eye_indices)
        right_eye_pts = self._get_eye_points(self.right_eye_indices)
        ear = (self._calculate_ear(left_eye_pts) + self._calculate_ear(right_eye_pts)) / 2.0
        st = "CLOSED" if ear < self.ear_threshold and self.frame_counter>=self.consec_frames else "OPEN"
        clr = (0,0,255) if st=="CLOSED" else (0,255,0)
        cv2.putText(frame, f"EYES {st}", (50,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, clr,2)
        y += 40

        # gaze
        gaze_res = self._detect_gaze_direction()
        if isinstance(gaze_res, tuple):
            gd, gr = gaze_res
            self.gaze_history.append((t, gd, gr))
        else:
            gd, gr = gaze_res, np.nan
            self.gaze_history.append((t, gd, gr))
        clr = {"LEFT":(0,165,255),"RIGHT":(0,165,255),"CENTER":(0,255,0),
               "NO FACE":(0,0,255),"UNKNOWN":(128,128,128)}.get(gd,(255,255,255))
        cv2.putText(frame, f"LOOKING: {gd}", (50,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, clr,2)
        y += 30

        # movement labels
        if h_move:
            cv2.putText(frame, f"H_MOVE: {h_dir}", (50,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)
            y += 30
        if v_move:
            cv2.putText(frame, f"V_MOVE: {v_dir}", (50,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)

        return frame


if __name__ == "__main__":
    cap = cv2.VideoCapture("video_recordings/alessandro.mov")  # or 0 for webcam
    gaze = MediaPipeGazeTracking()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_count / fps
        frame_count += 1

        gaze.refresh(frame)
        out = gaze.annotated_frame(t)

        cv2.imshow("Gaze Tracking", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # save logs
    gaze.export_to_csv()
    gaze.export_gaze_to_csv()
    gaze.export_movement_to_csv()