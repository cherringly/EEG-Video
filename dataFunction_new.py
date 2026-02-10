'''
🛠 Hardware Configuration Sync 

| Parameter | Code Variable | Current Value |
| :--- | :--- | :--- |
| **UDP Port** | `self.udp_port` | `9000` |
| **Sampling Rate** | `self.fs` | `5537 Hz` |
| **ADC Bits** | `self.ADCres` | `12 bits` |
| **Data Format** | `dtype` | `np.int16` |
| **Reference Voltage**| `Vref` | `1.8 V` |
'''
from scipy.signal import butter, lfilter, lfilter_zi
import time
import threading
import queue
import socket

class BionodeProcessor:
    def __init__(self, ADCres=12, sampR=5537, numChannels=4, filter_config=None):
        """
        Initialize the real-time processor.
        """
        self.ADCres = ADCres
        self.fs = sampR
        self.numChannels = numChannels
        self.filter_config = filter_config

        # --- DEVICE SPECIFIC CONFIGURATION START ---
        # Set to "0.0.0.0" to listen on all interfaces.
        self.udp_ip = "0.0.0.0"  
        # This port must match the destination port in  BioNode device (e.g., 9000).
        self.udp_port = 9000
        # --- DEVICE SPECIFIC CONFIGURATION END ---

        # Initialize the UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))

        # Setup Filters with State (zi)
        self.filters = {}
        if filter_config:
            for ch, config in filter_config.items():
                if config["type"] == "low":
                    b, a = butter(4, config["cutoff"] / (self.fs / 2), btype='low')
                elif config["type"] == "high":
                    b, a = butter(4, config["cutoff"] / (self.fs / 2), btype='high')
                elif config["type"] == "bandpass":
                    low, high = config["cutoff"]
                    b, a = butter(4, [low / (self.fs / 2), high / (self.fs / 2)], btype='bandpass')
                else:
                    continue
                
                # Initialize filter state for seamless real-time processing
                zi = lfilter_zi(b, a)
                self.filters[ch] = {'b': b, 'a': a, 'zi': zi}

        self.data_queue = queue.Queue(maxsize=100)
        self.running = False
        self.background_thread = None

    def unpack_bionode_packet(self, data):
        """
        Parses raw binary UDP bytes into a structured NumPy array.
        Modify 'dtype' and 'reshape' based on your actual data packet structure.
        """
        try:
            # Converting raw buffer to 16-bit integers 
            # - Use np.int16: If ADC is 12-bit or 16-bit (2 bytes per sample).
            # - Use np.int32: If ADC is 24-bit (usually padded to 4 bytes per sample).
            raw_array = np.frombuffer(data, dtype=np.int16)
        
            # Reshape to (Channels, Samples). Adjust based on packet size.
            # Example: 4 channels, and whatever samples fit in the packet.
            return raw_array.reshape(self.numChannels, -1)
        except ValueError as e:
            print(f"[ERROR] Packet size mismatch in unpack_bionode_packet: {e}")
            return np.zeros((self.numChannels, 1)) # Return empty chunk to prevent crash
        
    def process_chunk(self, raw_chunk):
        """
        Main processing pipeline: Scaling (to mV) followed by Causal Filtering.
        """
        raw = raw_chunk.astype(np.float32)
        offset = 2**(self.ADCres - 1)
        denom = 2**self.ADCres
        
        # CORRECTED: Map to Voltage (1.8V Ref) and convert to Millivolts (* 1000)
        scaled = (raw - offset) * 1.8 / denom * 1000 

        filtered_data = np.zeros_like(scaled)
        for ch in range(self.numChannels):
            if self.filter_config and ch in self.filters:
                f = self.filters[ch]
                # Apply filter and update state 'zi' for next chunk
                y, next_zi = lfilter(f['b'], f['a'], scaled[ch], zi=f['zi'])
                filtered_data[ch] = y
                self.filters[ch]['zi'] = next_zi 
            else:
                filtered_data[ch] = scaled[ch]
                
        return filtered_data

    def _data_acquisition_loop(self):
        """
        Hardware-driven acquisition loop.
        """
        while self.running:
            try:
                # Blocking read: Waits for hardware to push a packet
                data, addr = self.sock.recvfrom(2048) 
                
                # Unpack bytes to numerical array
                raw_chunk = self.unpack_bionode_packet(data) 
                
                if not self.data_queue.full():
                    self.data_queue.put(raw_chunk)
                
            except Exception as e:
                print(f"UDP Error: {e}")
                break

    def start_bionode_stream(self):
        self.running = True
        self.background_thread = threading.Thread(target=self._data_acquisition_loop, daemon=True)
        self.background_thread.start()
        print(f"[INFO] Listening for BioNode data on port {self.udp_port}...")

    def stop(self):
        self.running = False
        if self.background_thread:
            self.background_thread.join()
        self.sock.close()