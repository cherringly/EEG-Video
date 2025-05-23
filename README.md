This suite allows for a quick calibration of BCI devices using a computer camera. The code uses object recognition and signal processing to recognize and validate presence of EMG, EOG and EEG in the signal, determining quickly level of readiness of the device.


## integrated_new.py
### PURPOSE: 
synchronizes EEG recordings and gaze tracking to visualize and analyze correlations between eye movements and brain activity in real time

| Variable                        | Purpose                                                                   |
| ------------------------------- | ------------------------------------------------------------------------- |
| `rawCha`                        | Raw EEG values (converted from ADC)                                       |
| `filtered`                      | EEG after low-pass filter (removes high-freq noise)                       |
| `paused`                        | Flag to pause the animation (shared between threads)                      |
| `video_frame_time`              | Shared float list holding current video time                              |
| `queue_frame`                   | Thread-safe queue for passing frames between video thread and plot thread |
| `alpha_power`, `smoothed_power` | Alpha power (raw and smoothed) for visualization                          |
| `text_labels`                   | List of dynamic annotations showing eye movement detection                |

### MAIN FUNCTIONS:
run_video()
Runs in a background thread to
Read and resize video frames.
Track gaze and blink events.
Annotate frames and update a queue.
Save gaze data to CSV when the video ends.

detect_eye_movements(y_win, t_win)
A simple spike-dip heuristic to detect eye movement artifacts in the EEG trace:
Detects a large upward spike followed quickly by a downward dip.
Used to label saccadic movements or potential EOG noise.

update(frame_num)
Runs on every frame refresh to:
Synchronize EEG and video displays using a shared timeline (video_frame_time).
Update EEG voltage plot and mark detected eye movements.
Display updated alpha power.
Pull latest annotated video frame from the queue_frame.


### DEPENDENCIES:
- numpy
- matplotlib (pyplot and animation)
- cv2
- scipy
- threading
- bionodebinopen.py
- parallel.py
- gaze_track.py



