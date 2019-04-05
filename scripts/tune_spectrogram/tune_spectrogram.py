"""
This script takes an audio file and turns it into a series of spectrograms that you can
look at and see if they look good or not.

This is useful for tuning the spectrogram providing function to provide spectrograms
that are likely to be the right dimensions, windowing function, etc. to make learning
useful features from them easier.
"""
import audiosegment as asg
import matplotlib.pyplot as plt
import numpy as np
import os
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
    sample_rate_hz  = 8000.0    # 16kHz sample rate
    bytewidth       = 2          # 16-bit samples
    nchannels       = 1          # mono

    # Spectrogram
    duration_s      = 0.30        # Duration of each complete spectrogram
    window_length_s = 0.020       # How long each FFT is
    overlap         = 0.2        # How much each FFT overlaps with each other one

    # Track
    first_start_s   = 14          # Where in the track should we start grabbing spectrograms?
    #################################################################

    # Load the audio file into an AudioSegment
    seg = asg.from_file(sys.argv[1])
    seg = seg.resample(sample_rate_Hz=sample_rate_hz, sample_width=bytewidth, channels=nchannels)

    nspectrograms = 4
    for i in range(1, nspectrograms + 1):
        idx = i - 1
        start_s = idx * duration_s + first_start_s
        frequencies, times, amplitudes = seg.spectrogram(start_s, duration_s, window_length_s=window_length_s, overlap=overlap)
        print("Fs, ts, amps:", frequencies.shape, times.shape, amplitudes.shape)

        # Log the amplitudes to help with contrast
        #amplitudes = 10 * np.log10(amplitudes + 1e-9)

        # Plot into a subplot
        subplotidx = 100 + (nspectrograms * 10) + i
        plt.subplot(subplotidx)
        plt.pcolormesh(times, frequencies, amplitudes)

    ## Now finally show the spectrograms
    plt.show()
