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
import output.voice.synthesizer as synth                        # pylint: disable=locally-disabled, import-error


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
    [0.3, 0.2, 0.5],
    # Genioglossus      ###### Seems to only zero stuff out or do nothing. Set to zero.
    [0.0, 0.0, 0.0],
    # UpperTongue       ##### Values below zero seem to do things
    [0.0, 0.0, 0.0],
    # LowerTongue       ##### Values below zero seem to do things
    [0.0, 0.0, 0.0],
    # TransverseTongue  ##### Can't get it to do anything
    [0.0, 0.0, 0.0],
    # VerticalTongue    ##### Can't get it to do anything
    [0.0, 0.0, 0.0],
    #Risorius           ##### Doesn't seem to do anything
    [0.0, 0.0, 0.0],
    #OrbicularisOris    #### 0.5 to 1.0
    [0.9, 0.9, 0.9],

    #LevatorPalatini ###### Full range, but hard to get it to do much
    [0.0, 0.0, 0.0],
    #TensorPalatini  ###### Ditto
    [0.0, 0.0, 0.0],

    #Masseter         ##### -0.5 to 0.0 (below zero seems to close the mouth a bit)
    [-0.5, -0.3, -0.1],
    #Mylohyoid        ##### Full range
    [0.0, 0.0, 0.0],
    #LateralPterygoid ##### Doesn't do much Set to Zeros
    [0.0, 0.0, 0.0],
    #Buccinator       ##### Doesn't do much
    [0.0, 0.0, 0.0],
])

if __name__ == "__main__":
    seg = synth.make_seg_from_synthmat(synthmat, duration_s, time_points_s)
    seg = seg.resample(sample_rate_Hz=16000.0, sample_width=2, channels=1)
    plt.plot(seg.to_numpy_array().astype(float))
    plt.show()
    seg.export("output.wav", format="WAV")
    print("Human audible?", seg.human_audible())
    print("RMS:", seg.rms)
    print("SPL:", seg.spl)
