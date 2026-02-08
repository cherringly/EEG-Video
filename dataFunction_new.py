import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
import time

class load_and_filter:
    def __init__(self, ADCres=12, sampR=5537, numChannels=4, filter_config=None):
        """
        Initialize the real-time processor.
        
        Args:
            ADCres (int): ADC resolution (12 or 24).
            sampR (int): Sampling rate in Hz.
            numChannels (int): Number of channels.
            filter_config (dict): Configuration for filters.
        """
        self.ADCres = ADCres
        self.fs = sampR
        self.numChannels = numChannels
        self.filter_config = filter_config
        
        # Core: Store filter coefficients (b, a) and real-time state (zi) for each channel
        self.filters = {}
        
        if filter_config:
            for ch, config in filter_config.items():
                # 1. Calculate Butterworth filter coefficients (b, a)
                if config["type"] == "low":
                    b, a = butter(4, config["cutoff"] / (self.fs / 2), btype='low')
                elif config["type"] == "high":
                    b, a = butter(4, config["cutoff"] / (self.fs / 2), btype='high')
                elif config["type"] == "bandpass":
                    low, high = config["cutoff"]
                    b, a = butter(4, [low / (self.fs / 2), high / (self.fs / 2)], btype='bandpass')
                else:
                    continue
                
                # 2. Initialize filter state (zi)
                # This is crucial for real-time filtering to ensure smooth transitions between data chunks
                zi = lfilter_zi(b, a)
                self.filters[ch] = {'b': b, 'a': a, 'zi': zi}

    def process_chunk(self, raw_chunk):
        """
        Process real-time data chunks.
        
        Args:
            raw_chunk (numpy.ndarray): Raw data, shape should be (numChannels, numSamples)
            
        Returns:
            numpy.ndarray: Scaled and filtered voltage data
        """
        # Step 1: Physical scaling (Logic retained from original load_and_filter)
        raw = raw_chunk.astype(np.float32)
        if self.ADCres == 12:
            scaled = (raw - 2**11) * 1.8 / (2**12 * 1000)
        elif self.ADCres == 24:
            scaled = (raw - 2**23) * 1.8 / (2**24 * 1000)
        else:
            scaled = raw

        # Step 2: Real-time filtering per channel
        filtered_data = np.zeros_like(scaled)
        for ch in range(self.numChannels):
            if self.filter_config and ch in self.filters:
                f = self.filters[ch]
                # Use lfilter (one-way) with the previous state zi
                # This replaces filtfilt (two-way) used in offline processing
                y, next_zi = lfilter(f['b'], f['a'], scaled[ch], zi=f['zi'])
                filtered_data[ch] = y
                # Update state for the next data chunk
                self.filters[ch]['zi'] = next_zi 
            else:
                filtered_data[ch] = scaled[ch]
                
        return filtered_data

def sync_to_wall_clock(vid_time, t0_real):
    """
    Synchronize processing pace with real-world time.
    """
    if t0_real[0] is None:
        t0_real[0] = time.time() - vid_time
    to_sleep = t0_real[0] + vid_time - time.time()
    if to_sleep > 0:
        time.sleep(to_sleep)
    return to_sleep