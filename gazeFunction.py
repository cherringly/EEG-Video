import time
import numpy as np
import cv2
from queue import Full
import mediapipe as mp
from active.gaze_track import MediaPipeGazeTracking
from dataFunction import sync_to_wall_clock

def gazeTrack(video_path, eeg_queue, paused, saccade_times, prev_direction, t0_real, eye_box_size=100):
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
    cap       = cv2.VideoCapture(video_path)
    gaze      = MediaPipeGazeTracking()
    start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    while True:
        if paused[0]:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        # timestamp & sync
        msec     = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        vid_time = msec - start_time
        sync_to_wall_clock(vid_time, t0_real)

        # update gaze tracker
        gaze.refresh(frame)
        blink     = gaze.is_blinking(vid_time)
        annotated = gaze.annotated_frame(vid_time)
        main_disp = cv2.resize(annotated, (320, 240))

        # always crop the eye region (even if blinking)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm       = res.multi_face_landmarks[0].landmark
            h, w, _  = frame.shape
            lx, ly   = int(lm[33].x * w), int(lm[33].y * h)
            rx, ry   = int(lm[263].x * w), int(lm[263].y * h)
            cx, cy   = (lx + rx)//2, (ly + ry)//2
            x1, x2   = max(cx - eye_box_size, 0), min(cx + eye_box_size, w)
            y1, y2   = max(cy - eye_box_size, 0), min(cy + eye_box_size, h)
            cropped  = annotated[y1:y2, x1:x2]
            zoom_disp = cv2.resize(cropped, (240, 240))
        else:
            zoom_disp = np.zeros((240, 240, 3), dtype=np.uint8)

        # only do saccade detection if eyes are open
        if not blink and res.multi_face_landmarks:
            d = gaze.gaze_direction
            if prev_direction[0] and d != prev_direction[0]:
                saccade_times.append(vid_time)
            prev_direction[0] = d

        # push to EEG queue regardless, so plotting stays in time
        try:
            eeg_queue.put_nowait((main_disp, zoom_disp, vid_time))
        except Full:
            pass

    cap.release()
