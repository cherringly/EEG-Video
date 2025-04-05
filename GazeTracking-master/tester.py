import cv2
from gaze_tracking import GazeTracking
import os

def detect_blinks_from_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

    gaze = GazeTracking()
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise RuntimeError("Failed to open video file.")

    print("Analyzing video for blinks...\n")

    while True:
        ret, frame = video.read()
        if not ret:
            break  # End of video

        gaze.refresh(frame)
        annotated_frame = gaze.annotated_frame()

        text = ""
        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        cv2.putText(annotated_frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(annotated_frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(annotated_frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

        cv2.imshow("Blink Detection", annotated_frame)

        if cv2.waitKey(1) == 27:  # ESC key to stop
            break

    video.release()
    cv2.destroyAllWindows()
    print("\nBlink analysis complete.")

if __name__ == "__main__":
    # Replace with your actual video path or accept via input/argparse
    test_video_path = "example.mov"
    detect_blinks_from_video(test_video_path)
