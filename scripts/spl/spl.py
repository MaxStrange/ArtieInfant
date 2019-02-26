import audiosegment
import sys

import math
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need a path to a WAV file.")
        exit(1)

    seg = audiosegment.from_file(sys.argv[1])
    print("Reward for this WAV file:", seg.rms)

    plt.title("Raw Values")
    plt.plot(seg.to_numpy_array())
    plt.show()

    plt.title("Histogram")
    hist_bins, hist_vals = seg.fft()
    hist_vals_real_normed = np.abs(hist_vals) / len(hist_vals)
    plt.plot(hist_bins/1000, hist_vals_real_normed)
    plt.xlabel("kHz")
    plt.ylabel("dB")
    plt.show()
