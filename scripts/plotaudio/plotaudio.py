"""
This is code that I find I use a LOT while debugging or analyzing.
"""
import audiosegment
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


#################################################
#### These are the parameters I have been using #
#################################################
# ---- for long spectrograms ------
sample_rate_hz  = 16000.0    # 16kHz sample rate
bytewidth       = 2          # 16-bit samples
nchannels       = 1          # mono
duration_s      = 0.5        # Duration of each complete spectrogram
window_length_s = 0.03       # How long each FFT is
overlap         = 0.2        # How much each FFT overlaps with each other one

# ---- for short spectrograms ------
#sample_rate_hz  = 8000.0    # 8kHz sample rate
#bytewidth       = 2          # 16-bit samples
#nchannels       = 1          # mono
#duration_s      = 0.3        # Duration of each complete spectrogram
#window_length_s = 0.02       # How long each FFT is
#overlap         = 0.2        # How much each FFT overlaps with each other one
#################################################

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need a path to a WAV file.")
        exit(1)

    seg = audiosegment.from_file(sys.argv[1])
    print(seg)
    print("  -> RMS:", seg.rms)
    print("  -> SPL:", seg.spl)
    print("  -> Length (s):", seg.duration_seconds)
    print("  -> NChannels:", seg.channels)
    print("  -> Frequency (Hz):", seg.frame_rate)
    print("  -> Bytes per sample:", seg.sample_width)
    print("  -> Human audible?", seg.human_audible())

    name = os.path.basename(sys.argv[1])
    name, _ext = os.path.splitext(name)

    plt.title("Raw Values")
    arr = seg.to_numpy_array()
    times = np.linspace(0, len(arr) / seg.frame_rate, num=len(arr))
    plt.plot(times, seg.to_numpy_array())
    plt.xlabel("Time (s)")
    plt.ylabel("PCM")
    plt.savefig("{}-waveform.png".format(name))
    plt.show()

    plt.title("Histogram")
    hist_bins, hist_vals = seg.fft()
    hist_vals_real_normed = np.abs(hist_vals) / len(hist_vals)
    plt.plot(hist_bins/1000, hist_vals_real_normed)
    plt.xlabel("kHz")
    plt.ylabel("dB")
    plt.savefig("{}-histogram.png".format(name))
    plt.show()

    plt.title("Spectrogram")
    fs, ts, amps = seg.spectrogram(0, duration_s, window_length_s=window_length_s, overlap=overlap, window=('tukey', 0.5))
    #amps = 10.0 * np.log10(amps)
    plt.pcolormesh(ts, fs, amps)
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.savefig("{}-spectrogram.png".format(name))
    plt.show()
