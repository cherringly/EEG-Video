from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def fn_BionodeBinOpen(packedFileDir: str, ADCres: int, sampR: int) -> dict:
    """
    Open a Bionode-generated .bin file, parse header and unpack 12-bit ADC samples
    into a channels×time array.

    Args:
        packedFileDir: path to the .bin file
        ADCres: ADC resolution in bits (e.g. 24)
        sampR: sampling rate (Hz) for the time vector

    Returns:
        dict with keys:
          - 'Date': datetime object from file header
          - 'sampleRate': sample rate from header (int)
          - 'numChannels': number of channels (int)
          - 'channelsData': ndarray, shape (num_channels, total_samples)
          - 'time': ndarray, shape (total_samples,)
    """
    # Read entire file as uint8
    try:
        with open(packedFileDir, 'rb') as f:
            raw_bytes = f.read()
    except IOError as e:
        raise IOError(f"Could not open file {packedFileDir}: {e}")

    raw_data   = np.frombuffer(raw_bytes, dtype=np.uint8)
    packet_len = 58
    packet_num = len(raw_data) // packet_len
    if packet_num < 2:
        raise ValueError("File contains no data packets.")

    # --- Parse header packet (first 58 bytes) ---
    hdr = raw_data[:packet_len]
    year        = (hdr[0] << 8) + hdr[1]
    month       = int(hdr[2])
    day         = int(hdr[3])
    hour        = int(hdr[4])
    minute      = int(hdr[5])
    second      = int(hdr[6])
    sampleRate  = (hdr[7] << 8) + hdr[8]
    numChannels = int(hdr[9])
    file_date   = datetime(year, month, day, hour, minute, second)

    # Determine how many 12-bit samples per channel per packet
    n_samps_per_pkt = 24 // numChannels
    n_data_pkts     = packet_num - 1
    total_samps     = n_data_pkts * n_samps_per_pkt

    # Preallocate channels × time array
    channelsData = np.zeros((numChannels, total_samps), dtype=np.uint16)

    print("Unpacking Data from file...")

    # Loop over data packets
    for i in range(1, packet_num):
        start = i * packet_len
        end   = start + packet_len
        pkt   = raw_data[start:end]

        # Extract the 3-byte groups: bytes 8 through (58-15)  → indices [7:43]
        x = pkt[7:packet_len-15]
        x = x.reshape(-1, 3)

        # Combine into 24-bit words
        y = (
            (x[:, 0].astype(np.uint32) << 16) |
            (x[:, 1].astype(np.uint32) <<  8) |
             x[:, 2].astype(np.uint32)
        )

        # Split each into two 12-bit samples
        high = (y & 0xFFF000) >> 12
        low  =  y & 0x000FFF
        interleaved = np.column_stack((high, low)).ravel(order='F')

        # Reshape into (channels × samples_per_chunk)
        pkt_data = interleaved.reshape((numChannels, n_samps_per_pkt), order='F')

        idx0 = (i-1) * n_samps_per_pkt
        idx1 = idx0 + n_samps_per_pkt
        channelsData[:, idx0:idx1] = pkt_data

        # Progress update every ~5%
        if i % max(1, round(packet_num * 0.05)) == 0:
            print(f"Progress: {i/packet_num*100:.0f}%")

    print("Unpacking Completed!")

    # Build the time vector
    time = np.arange(total_samps) / sampR

    return {
        'Date':         file_date,
        'sampleRate':   sampleRate,
        'numChannels':  numChannels,
        'channelsData': channelsData,
        'time':         time
    }


def main():
    # Input parameters
    packedFileDir = "ear3.31.25_1.bin"
    ADCres        = 24       # ADC resolution (bits)
    sampR         = 1000     # Sampling rate in Hz

    # Load and unpack data
    unpacked = fn_BionodeBinOpen(packedFileDir, ADCres, sampR)

    print("\n--- File Info ---")
    print("Date:       ", unpacked["Date"])
    print("SampleRate: ", unpacked["sampleRate"])
    print("NumChannels:", unpacked["numChannels"])
    print("Shape:      ", unpacked["channelsData"].shape)

    # Quick preview of channel 0
    plt.plot(unpacked["time"], unpacked["channelsData"][0])
    plt.title("Channel 0 Preview")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


if __name__ == "__main__":
    main()
