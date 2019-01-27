"""
This script is useful for testing ways to save spectrograms as images.
"""
import audiosegment as asg
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need path to test audio file")
        exit(1)
    elif not os.path.isfile(sys.argv[1]):
        print("Need a valid path")
        exit(2)

    #################################################################
    # Here are the configurations that you should tune to your liking
    #################################################################
    # Resampling
    sample_rate_hz  = 16000.0    # 16kHz sample rate
    bytewidth       = 2          # 16-bit samples
    nchannels       = 1          # mono

    # Spectrogram
    duration_s      = 0.5        # Duration of each complete spectrogram
    window_length_s = 0.03       # How long each FFT is
    overlap         = 0.2        # How much each FFT overlaps with each other one

    # Track
    start_s         = 0          # Where in the track should we start grabbing spectrograms?
    #################################################################

    # Load the audio file into an AudioSegment
    seg = asg.from_file(sys.argv[1])
    seg = seg.resample(sample_rate_Hz=sample_rate_hz, sample_width=bytewidth, channels=nchannels)
    #filtered, filterbankfreqs = seg.filter_bank(lower_bound_hz=50, upper_bound_hz=4000, nfilters=128, mode='mel')
    #seg = asg.from_numpy_array(np.sum(filtered, axis=0, dtype=np.uint16), framerate=sample_rate_hz)
    fs, ts, amps = seg.spectrogram(start_s, duration_s, window_length_s=window_length_s, overlap=overlap, window=('tukey', 0.25))
    #amps = 10.0 * np.log10(amps +1e-9)

    plt.subplot(121)
    plt.pcolormesh(ts, fs, amps)

    amps *= 255.0 / np.max(amps)
    amps = amps.astype(np.uint8)

    plt.subplot(122)
    plt.pcolormesh(ts, fs, amps)

    plt.show()

    print(amps.shape)
    imageio.imwrite("outputimage_imwrite.png", amps)
    amps = imageio.imread("outputimage_imwrite.png")
    print(amps.shape)
