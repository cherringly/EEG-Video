import cv2
from gaze_tracking import GazeTracking
import os

class BlinkDetector:
    def __init__(self):
        self.last_state = None
        self.eeg_text = ("EEG X", (0, 255, 0))  # (text, color)
        self.eog_text = ("EOG", (0, 0, 0))
        
    def update_blink_state(self, frame, is_blinking):
        if is_blinking:
            self.eeg_text = ("EEG", (0, 0, 0))
            self.eog_text = ("EOG X", (0, 0, 255))
            state_changed = True
        else:
            self.eeg_text = ("EEG X", (0, 255, 0))
            self.eog_text = ("EOG", (0, 0, 0))
            state_changed = (self.last_state == True)
        
        self.last_state = is_blinking
        return state_changed

def detect_blinks_from_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file '{video_path}' does not exist.")

    gaze = GazeTracking()
    detector = BlinkDetector()
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps) if fps > 0 else 33

    if not video.isOpened():
        raise RuntimeError("Failed to open video file.")

    print("Analyzing video for blinks... Press ESC to quit\n")

    # Create a base frame with static elements
    ret, base_frame = video.read()
    if not ret:
        video.release()
        return

    # Pre-draw static elements
    cv2.putText(base_frame, "EMG", (90, 260), cv2.FONT_HERSHEY_DUPLEX, 3.2, (0,0,0), 2)
    cv2.putText(base_frame, "ECG", (90, 350), cv2.FONT_HERSHEY_DUPLEX, 3.2, (0,0,0), 2)
    static_elements = base_frame.copy()

    while True:
        ret, frame = video.read()
        frame_small =  cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if not ret:
            break

        # Process the frame with GazeTracking
        gaze.refresh(frame_small)
        is_blinking = not gaze.is_center()
        
        # Only update dynamic elements when state changes
        frame_small = static_elements.copy()
        if detector.update_blink_state(frame_small, is_blinking):
            cv2.putText(frame_small, detector.eeg_text[0], (90, 80), 
                       cv2.FONT_HERSHEY_DUPLEX, 3.2, detector.eeg_text[1], 2)
            cv2.putText(frame_small, detector.eog_text[0], (90, 170), 
                       cv2.FONT_HERSHEY_DUPLEX, 3.2, detector.eog_text[1], 2)

        # Display the frame
        cv2.imshow("Blink Detection", frame_small)

        # Exit when ESC key is pressed
        if cv2.waitKey(frame_delay) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()
    print("\nBlink analysis complete.")



if __name__ == "__main__":
    test_video_path = "3.31.25_1.mov"
    detect_blinks_from_video(test_video_path)