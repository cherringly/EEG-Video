from datetime import datetime
import numpy as np


def fn_BionodeBinOpen(packedFileDir: str, ADCres: int, sampR: int) -> dict:
    unpackedFile = {}

    try:
        with open(packedFileDir, 'rb') as f:
            rawData = np.frombuffer(f.read(), dtype=np.uint8)
    except FileNotFoundError:
        raise FileNotFoundError("File could not be opened. Check the path or permissions.")

    # Read header
    year = (rawData[0] << 8) + rawData[1]
    month = rawData[2]
    day = rawData[3]
    hour = rawData[4]
    minute = rawData[5]
    second = rawData[6]
    sampleRate = (rawData[7] << 8) + rawData[8]
    numChannels = rawData[9]

    unpackedFile['Date'] = datetime(year, month, day, hour, minute, second)
    unpackedFile['sampleRate'] = sampleRate
    unpackedFile['numChannels'] = numChannels

    bytesPerSample = ADCres // 8
    packet_size = 58
    packetNum = len(rawData) // packet_size
    channelsData = np.zeros((numChannels, 13*(packetNum - 1)), dtype=np.uint32)  # Adjust dtype as needed
    print("Unpacking Data from file... ")
    print(channelsData.dtype)
    for i in range(1, packetNum):
        tempPacket = rawData[(packet_size * i):(packet_size * (i + 1))]
        X = tempPacket[7:-11]  # 8:end-11 in 0-based indexing

        yarr = np.array([
            (int(X[3*j]) << 16) | (int(X[3*j + 1]) << 8) | int(X[3*j + 2])
            for j in range(len(X) // 3)
        ], dtype=np.uint32)

        zarr = np.zeros(2 * len(yarr), dtype=np.uint16)
        for j, y in enumerate(yarr):
            z1 = (y & 0xFFF000) >> 12
            z2 = y & 0x000FFF
            zarr[2 * j] = z1
            zarr[2 * j + 1] = z2


        for idx, val in enumerate(zarr):
            channel = int(idx % numChannels)
            pos = len(yarr) * (i - 1) + int(np.ceil((idx + 1) / numChannels)) - 1
            if pos < channelsData.shape[1]:
                channelsData[channel, pos] = int(val)


        if i % max(1, packetNum // 20) == 0:
            print(f"Progress: {int((i / packetNum) * 100)}%")

    print("Unpacking Completed!")

    unpackedFile['channelsData'] = channelsData
    unpackedFile['time'] = np.arange(0, channelsData.shape[1]) / sampR

    return unpackedFile

def main():
    # Input parameters
    packedFileDir = "ear3.31.25_1.bin"
    ADCres = 24       # ADC resolution (e.g., 24 bits)
    sampR = 1000      # Sampling rate in Hz (update as needed)

    # Load and unpack data
    unpacked = fn_BionodeBinOpen(packedFileDir, ADCres, sampR)

    print("\n--- File Info ---")
    print("Date:", unpacked["Date"])
    print("Sample Rate:", unpacked["sampleRate"])
    print("Channels:", unpacked["numChannels"])
    print("Shape:", unpacked["channelsData"].shape)

    # Optional: Plot a quick preview
    import matplotlib.pyplot as plt
    plt.plot(unpacked["time"], unpacked["channelsData"][0])
    plt.title("Channel 0 Preview")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


if __name__ == "__main__":
    main()
