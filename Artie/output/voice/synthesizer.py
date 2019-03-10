"""
This script generates a random set of parameters to use in
Praat's articulatory synthesizer.
"""
import audiosegment
import logging
import numpy as np
import os
import pandas
import random
import string
import subprocess
import sys
import tempfile

# These are the 'muscles' involved in the production of speech (at least
# as far as is used by Praat's articulatory synthesizer)
articularizers = [
    'Lungs',                    # 0  # Laryngeal
    'Interarytenoid',           # 1  # Laryngeal
    'Cricothyroid',             # 2  # Laryngeal
    'Vocalis',                  # 3  # Laryngeal
    'Thyroarytenoid',           # 4  # Laryngeal
    'PosteriorCricoarytenoid',  # 5  # Laryngeal
    'LateralCricoarytenoid',    # 6  # Laryngeal
    'Stylohyoid',               # 7  # Laryngeal
    'Thyropharyngeus',          # 8  # Swallowing
    'LowerConstrictor',         # 9  # Swallowing
    'MiddleConstrictor',        # 10 # Swallowing
    'UpperConstrictor',         # 11 # Swallowing
    'Sphincter',                # 12 # Swallowing
    'Hyoglossus',               # 13 # Lingual
    'Styloglossus',             # 14 # Lingual
    'Genioglossus',             # 15 # Lingual
    'UpperTongue',              # 16 # Lingual
    'LowerTongue',              # 17 # Lingual
    'TransverseTongue',         # 18 # Lingual
    'VerticalTongue',           # 19 # Lingual
    'Risorius',                 # 20 # Labial
    'OrbicularisOris',          # 21 # Labial
    'LevatorPalatini',          # 22 # Nasal
    'TensorPalatini',           # 23 # Nasal
    'Masseter',                 # 24 # Jaw
    'Mylohyoid',                # 25 # Lingual
    'LateralPterygoid',         # 26 # Jaw
    'Buccinator'                # 27 # Labial
]

# This is a numpy array of the form [True, False, False, True, etc.], where
# each True means that the corresponding articulator is laryngeal.
laryngeal_articulator_mask = np.array([
    articularizers.index('Lungs'),
    articularizers.index('Interarytenoid'),
    articularizers.index('Cricothyroid'),
    articularizers.index('Vocalis'),
    articularizers.index('Thyroarytenoid'),
    articularizers.index('PosteriorCricoarytenoid'),
    articularizers.index('LateralCricoarytenoid'),
    articularizers.index('Stylohyoid'),
])

# Ditto, but for jaw muscles
jaw_articulator_mask = np.array([
    articularizers.index('Masseter'),
    articularizers.index('LateralPterygoid'),
])

# Ditto, but for nasalizer muscles
nasal_articulator_mask = np.array([
    articularizers.index('LevatorPalatini'),
    articularizers.index('TensorPalatini'),
])

# Ditto, but for labial (lip) muscles
labial_articulator_mask = np.array([
    articularizers.index('Risorius'),
    articularizers.index('OrbicularisOris'),
    articularizers.index('Buccinator'),
])

# Ditto, but for lingual muscles (muscles that help with the tongue)
lingual_articulator_mask = np.array([
    articularizers.index('Hyoglossus'),
    articularizers.index('Styloglossus'),
    articularizers.index('Genioglossus'),
    articularizers.index('UpperTongue'),
    articularizers.index('LowerTongue'),
    articularizers.index('TransverseTongue'),
    articularizers.index('VerticalTongue'),
    articularizers.index('Mylohyoid'),
])

# Ditto, but for just the muscles that support the tongue somehow
lingual_articulator_support_mask = np.array([
    articularizers.index('Hyoglossus'),
    articularizers.index('Styloglossus'),
    articularizers.index('Genioglossus'),
    articularizers.index('Mylohyoid'),
])

# Ditto, but for just the tongue muscles
lingual_articulator_tongue_mask = np.array([
    articularizers.index('UpperTongue'),
    articularizers.index('LowerTongue'),
    articularizers.index('TransverseTongue'),
    articularizers.index('VerticalTongue'),
])

# Ditto, but for swallowing muscles
swallowing_articulator_mask = np.array([
    articularizers.index('Thyropharyngeus'),
    articularizers.index('LowerConstrictor'),
    articularizers.index('MiddleConstrictor'),
    articularizers.index('UpperConstrictor'),
    articularizers.index('Sphincter'),
])


scripttemplate = """
; These are all the variables we will need
duration = {duration}
fpath$ = "{resultfname}"

; Make a speaker and an articulation
speaker = Create Speaker: "speaker", "Female", "2"
artword = Create Artword: "hallo", duration

; Adjust the artword to articulate what we want
;           time value
selectObject: artword
{articulatory_string}

; Select the objects and make the sound
selectObject: speaker
plusObject: artword
sound = To Sound: 22050, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0

; Save the sound
selectObject: sound
Save as WAV file: fpath$
"""

def _make_praat_script(synthmat, duration, times):
    """
    Creates a Praat script with the given articulatory synthesis info.
    """
    articulatory_string = ""
    for i, art in enumerate(articularizers):
        for value, time in zip(synthmat[i, :], times):
            s = "Set target: {time}, {value:0.2f}, \"{name}\"\n".format(
                time=time, value=value, name=art
            )
            articulatory_string += s
    scripttext = scripttemplate.format(
        duration=duration, resultfname="{resultfname}", articulatory_string=articulatory_string
    )
    logging.debug("Generated the following script for Praat: {}".format(scripttext))
    return scripttext

def _run_praat_script(script):
    """
    Creates a temporary Praat script file, then runs Praat with it. Returns the WAV file path generated.
    """
    # Get a random base name
    randfname = "".join(random.sample(string.ascii_letters, 15))

    # Fill in the resulting file name in the script text
    resultfname = randfname + ".wav"
    script = script.format(resultfname=resultfname)

    # Create the randomly named script
    fpath = randfname + ".praat"
    with open(fpath, 'w') as f:
        f.write(script)

    # Call Praat on it
    subprocess.call(["praat", "--run", fpath])

    # Remove the script
    os.remove(fpath)

    # Tell the caller what the name of the resulting file is
    return resultfname

def make_seg_from_synthmat(synthmat, duration, times):
    """
    Makes a WAV file using Praat's articulatory synthesis module. Creates a script, saves it, then
    runs it via Praat. Returns an AudioSegment representation of the WAV file, and cleans up files.

    An example synthmat looks like this (this synthmat makes a sound similar to "ubbah"):

    ```
    duration = 0.5  # half a second
    times =                  0.00  0.10  0.25  0.50

    Lungs                     0.2   0.0   0.0   0.0
    Interarytenoid            0.5   0.5   0.5   0.5
    Cricothyroid              0.0   0.0   0.0   0.0
    Vocalis                   0.0   0.0   0.0   0.0
    Thyroarytenoid            0.0   0.0   0.0   0.0
    PosteriorCricoarytenoid   0.0   0.0   0.0   0.0
    LateralCricoarytenoid     0.0   0.0   0.0   0.0
    Stylohyoid                0.0   0.0   0.0   0.0
    Thyropharyngeus           0.0   0.0   0.0   0.0
    LowerConstrictor          0.0   0.0   0.0   0.0
    MiddleConstrictor         0.0   0.0   0.0   0.0
    UpperConstrictor          0.0   0.0   0.0   0.0
    Sphincter                 0.0   0.0   0.0   0.0
    Hyoglossus                0.0   0.0   0.0   0.0
    Styloglossus              0.0   0.0   0.0   0.0
    Genioglossus              0.0   0.0   0.0   0.0
    UpperTongue               0.0   0.0   0.0   0.0
    LowerTongue               0.0   0.0   0.0   0.0
    TransverseTongue          0.0   0.0   0.0   0.0
    VerticalTongue            0.0   0.0   0.0   0.0
    Risorius                  0.0   0.0   0.0   0.0
    OrbicularisOris           0.0   0.0   0.2   0.0
    LevatorPalatini           1.0   1.0   1.0   1.0
    TensorPalatini            0.0   0.0   0.0   0.0
    Masseter                  0.0   0.0   0.7   0.0
    Mylohyoid                 0.0   0.0   0.0   0.0
    LateralPterygoid          0.0   0.0   0.0   0.0
    Buccinator                0.0   0.0   0.0   0.0
    ```
    :param synthmat:    Numpy array of shape (len(articularizers), ntime_samples) that shows the activity
                        of each articularizer at each time point (Praat will linearly interpolate the activity
                        between the time points as appropriate).
    :param duration:    The total duration of the sound in seconds.
    :param times:       The times at which the values change.
    :returns:           An AudioSegment object, created from a WAV file output by Praat. The WAV file itself is
                        removed from disk.
    """
    assert len(synthmat.shape) == 2, "Wrong number of dimensions for synthmat. Expected 2, but got {}".format(len(synthmat.shape))
    for idx, time in enumerate(times):
        if time < 0 or time > duration:
            raise ValueError("Time", idx, "is invalid. Times must be non-negative and less than or equal to the duration, which is", duration)
    script = _make_praat_script(synthmat, duration, times)
    fpath = _run_praat_script(script)
    seg = audiosegment.from_file(fpath)
    os.remove(fpath)
    return seg

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: {} <n_time_points>".format(sys.argv[0]))
        print("Defaulting to trying with a known noise...")

    if len(sys.argv) == 2:
        # Parse out the nsamples
        try:
            nsamples = int(sys.argv[1])
            if nsamples <= 0:
                raise ValueError
        except ValueError:
            print("Need a positive integer for number of time points, but got", sys.argv[1])

        # Create the matrix
        synthmat = np.random.sample(size=(len(articularizers), nsamples))
        duration = 0.5  # half second duration
        times = np.linspace(0.0, duration, num=nsamples)
        synthmat[0, 0] = 0.2
        synthmat[0, int(nsamples/2):] = 0.0
    else:
        synthmat = np.zeros(shape=(len(articularizers), 4))
        synthmat[1, :] = 0.5  # Interarytenoid
        synthmat[22, :] = 1.0 # LevatorPalatini
        synthmat[0, 0] = 0.2  # Lungs
        synthmat[0, 1] = 0.0  # Lungs
        synthmat[24, 2] = 0.7 # Masseter
        synthmat[21, 2] = 0.2 # ObicularisOris
        duration = 0.5
        times = [0.0, 0.1, 0.25, 0.5]

    # Print it out in a useful format
    df = pandas.DataFrame(synthmat, index=articularizers, columns=times)
    print(df)

    # Now create a Praat Script that creates the produced sound
    seg = make_seg_from_synthmat(synthmat, duration=duration, times=times)
    seg.export("produced_sound.wav", format="WAV")
