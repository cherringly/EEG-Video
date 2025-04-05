import cv2
from gaze_tracking import GazeTracking
import os


def detect_blinks_from_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

    gaze = GazeTracking()
    video = cv2.VideoCapture(video_path)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps) if fps > 0 else 33

    if not video.isOpened():
        raise RuntimeError("Failed to open video file.")

    print("Analyzing video for blinks...\n")

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Process the frame with GazeTracking
        gaze.refresh(frame)
        frame = gaze.annotated_frame()

        if not (gaze.is_center()): #if blinking, then EOG
            label_eeg = "EEG"
            label_eog = "EOG X"
            cv2.putText(frame, label_eeg, (90, 80), cv2.FONT_HERSHEY_DUPLEX, 3.2, (0,0,0), 2)
            cv2.putText(frame, label_eog, (90, 170), cv2.FONT_HERSHEY_DUPLEX, 3.2, (0, 0, 255), 2)
        else: #if not blinking, then EEG
            label_eeg = "EEG X"
            label_eog = "EOG"
            cv2.putText(frame, label_eeg, (90, 80), cv2.FONT_HERSHEY_DUPLEX, 3.2, (0, 255, 0), 2) #bgr
            cv2.putText(frame, label_eog, (90, 170), cv2.FONT_HERSHEY_DUPLEX, 3.2, (0,0, 0), 2)
            
        cv2.putText(frame, "EMG", (90, 260), cv2.FONT_HERSHEY_DUPLEX, 3.2, (0,0, 0), 2)
        cv2.putText(frame, "ECG", (90, 350), cv2.FONT_HERSHEY_DUPLEX, 3.2, (0,0, 0), 2)

        # Overlay the label on the frame
        # cv2.putText(frame, label_eeg, (90, 80), cv2.FONT_HERSHEY_DUPLEX, 3.2, (255, 49, 49), 2)
        # cv2.putText(frame, label_eog, (90, 170), cv2.FONT_HERSHEY_DUPLEX, 3.2, (170, 255, 0), 2)

        # Display the frame
        cv2.imshow("Blink Detection", frame)

        # Exit when ESC key is pressed
        if cv2.waitKey(frame_delay) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()
    print("\nBlink analysis complete.")

if __name__ == "__main__":
    test_video_path = "3.31.25_1.mov"
    detect_blinks_from_video(test_video_path)
