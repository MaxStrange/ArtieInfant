"""
This script compares the spectrograms for two vowels side by side.
"""
import audiosegment as asg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need paths to two audio files.")
        exit(1)
    elif not os.path.isfile(sys.argv[1]):
        print("First argument must be a valid path.")
        exit(2)
    elif not os.path.isfile(sys.argv[2]):
        print("Second argument must be a valid path.")
        exit(3)

    ########################################################
    ## Parameters ##
    sample_rate_hz  = 16000.0
    sample_width    = 2
    nchannels       = 1

    window_length_s = 0.032
    overlap         = 0.20
    ########################################################

    seg1 = asg.from_file(sys.argv[1])
    seg2 = asg.from_file(sys.argv[2])

    seg1 = seg1.resample(sample_rate_Hz=sample_rate_hz, sample_width=sample_width, channels=nchannels)
    seg2 = seg2.resample(sample_rate_Hz=sample_rate_hz, sample_width=sample_width, channels=nchannels)

    print("--------------------------")
    print(seg1)
    print("--------------------------")
    print(seg2)
    print("--------------------------")

    # First spectrogram
    fs, ts, amps = seg1.spectrogram(window_length_s=window_length_s, overlap=overlap)
    amps = 10 * np.log10(amps + 1E-9)
    plt.subplot(121)
    plt.pcolormesh(ts, fs, amps)

    # Second spectrogram
    fs, ts, amps = seg2.spectrogram(window_length_s=window_length_s, overlap=overlap)
    amps = 10 * np.log10(amps + 1E-9)
    plt.subplot(122)
    plt.pcolormesh(ts, fs, amps)

    plt.show()
