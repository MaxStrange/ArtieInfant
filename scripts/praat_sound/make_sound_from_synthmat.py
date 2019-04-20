"""
This file provides a synthesis matrix and a function to make a sound from it.
Useful for checking what various modifications to the synthmat does to the output sound.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("../../Artie/experiment"))
sys.path.append(os.path.abspath("../../Artie"))
try:
    import output.voice.synthesizer as synth                        # pylint: disable=locally-disabled, import-error
except ImportError:
    print("Could not import stuff. Try running from inside this script's folder or fixing this janky hack.")
    exit(1)

#def make_seg_from_synthmat(synthmat, duration, times):
duration_s = 0.5
time_points_s = [0.0, 0.2, 0.4]
synthmat = np.array([
    # Lungs          ######## Leave at 0.2, 0.0, 0.0
    [0.2, 0.0, 0.0],
    # Interarytenoid ######## Leave at 0.5
    [0.5, 0.5, 0.5],
    # Cricothyroid   ######## Let this range over -1.0 to 1.0
    [-1.0, 0.0, 1.0],
    #Vocalis         ######## Let this range over -1.0 to 1.0
    [-1.0, 0.0, 1.0],
    # Thyroarytenoid ######## Let this range over -1.0 to 1.0
    [-1.0, 0.0, 1.0],
    # PosteriorCricoarytenoid ###### Keep this at 0.0
    [0.0, 0.0, 0.0],
    # LateralCricoarytenoid ######## Keep this at 0.0
    [0.0, 0.0, 0.0],

    # Group of muscles used in swallowing. Less so in speech. Set to zeros.
    # Stylohyoid
    [0.0, 0.0, 0.0],
    # Thyropharyngeus
    [0.0, 0.0, 0.0],
    # LowerConstrictor
    [0.0, 0.0, 0.0],
    # MiddleConstrictor
    [0.0, 0.0, 0.0],
    # UpperConstrictor
    [0.0, 0.0, 0.0],
    # Sphincter
    [0.0, 0.0, 0.0],

    # Hyoglossus        ###### Combines with other tongue stuff to modulate it. Don't bother. Set to zeros.
    [0.0, 0.0, 0.0],
    # Styloglossus      ###### Maxes out its usefulness at 0.5. Below 0.0 doesn't seem to do anything really. Above 0.5, silences.
    [0.0, 0.0, 0.5],
    # Genioglossus      ###### Seems to only zero stuff out or do nothing. Set to zero.
    [0.0, 0.0, 0.0],
    # UpperTongue       ##### Values below zero seem to do things
    [-1.0, -1.0, 0.0],
    # LowerTongue       ##### Values below zero seem to do things
    [0.0, 0.0, 0.0],
    # TransverseTongue  ##### Can't get it to do anything
    [0.0, 0.0, 0.0],
    # VerticalTongue    ##### Can't get it to do anything
    [0.0, 0.0, 0.0],
    #Risorius           ##### Doesn't seem to do anything
    [0.0, 0.0, 0.0],
    #OrbicularisOris    #### 0.5 to 1.0
    [0.7, 0.7, 0.5],

    #LevatorPalatini ###### Full range, but hard to get it to do much
    [0.0, 0.0, 0.0],
    #TensorPalatini  ###### Ditto
    [0.0, 0.0, 0.0],

    #Masseter         ##### -0.5 to 0.0 (below zero seems to close the mouth a bit)
    [-0.4, 0.0, -0.4],
    #Mylohyoid        ##### Full range
    [-0.8, 0.2, -0.8],
    #LateralPterygoid ##### Doesn't do much Set to Zeros
    [0.0, 0.0, 0.0],
    #Buccinator       ##### Doesn't do much
    [0.0, 0.0, 0.0],
])

if __name__ == "__main__":
    sample_rate_hz  = 16000.0    # 8kHz sample rate
    bytewidth       = 2          # 16-bit samples
    nchannels       = 1          # mono
    duration_s      = 0.5        # Duration of each complete spectrogram
    window_length_s = 0.03       # How long each FFT is
    overlap         = 0.2        # How much each FFT overlaps with each other one

    seg = synth.make_seg_from_synthmat(synthmat, duration_s, time_points_s)
    seg = seg.resample(sample_rate_Hz=sample_rate_hz, sample_width=bytewidth, channels=nchannels)
    plt.plot(seg.to_numpy_array().astype(float))
    plt.show()

    fs, ts, amps = seg.spectrogram(0, duration_s, window_length_s=window_length_s, overlap=overlap, window=('tukey', 0.5))
    plt.pcolormesh(ts, fs, amps)
    plt.ylabel("Hz")
    plt.xlabel("Time (s)")
    plt.show()

    seg.export("output.wav", format="WAV")
    print("Human audible?", seg.human_audible())
    print("RMS:", seg.rms)
    print("SPL:", seg.spl)
