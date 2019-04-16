"""
Analyzes the list of files hard-coded in this file to determine
if the xcor value of the files escape the variance of the xcor
of the random ones.
"""
import audiosegment as asg
import numpy as np
import os

# Location to prepend to all targets
baselocdir = "/home/max/Dropbox/thesis/results/real/geneticexperiments"

# Names of the experiments
experiment05_random = "geneticIII-0.5s-tps0.0-0.2-0.4-100pop-50gen-random-2px"
experiment03_random = "geneticIII-0.3s-tps0.0-0.1-0.25-100pop-50gen-random-2px"
experiment05_xcor   = "geneticI-0.5s-tps0.0-0.2-0.4-100pop-25gen-xcor-2px"
experiment03_xcor   = "geneticII-0.3s-tps0.0-0.1-0.25-100pop-25gen-xcor-2px"
experiment05_euclid = "closedloop-0.5s-tps0.0-0.2-0.4-100pop-12gen-euclid-2px"
experiment03_euclid = "closedloop-0.3s-tps0.0-0.1-0.25-100pop-12gen-euclid-2px"

# Names of the targets
target05_1 = "english_79.wav_33.wav"
target05_2 = "english_273.wav_33.wav"
target03_1 = "english_13313.wav_23.wav"
target03_2 = "english_14534.wav_10.wav"

halfsecond_directories = {
    "RANDOM": os.path.join(baselocdir, experiment05_random),
    "XCOR":   os.path.join(baselocdir, experiment05_xcor),
    "EUCLID": os.path.join(baselocdir, experiment05_euclid),
}

thirdsecond_directories = {
    "RANDOM": os.path.join(baselocdir, experiment03_random),
    "XCOR":   os.path.join(baselocdir, experiment03_xcor),
    "EUCLID": os.path.join(baselocdir, experiment03_euclid),
}

def xcorevaluate(seg: asg.AudioSegment, targetseg: asg.AudioSegment) -> float:
    """
    Applies the cross correlation procedure to the two segments and returns the value.
    """
    # Process the target sound
    target = targetseg.to_numpy_array().astype(float)
    target += abs(min(target))
    if max(target) != min(target):
        target /= max(target) - min(target)
    assert sum(target[target < 0]) == 0, "There are negative values after normalization! Error!"

    # Shift the wave form up by most negative value and normalize into 0.0 to 1.0
    ours = seg.to_numpy_array().astype(float)
    most_neg_val = min(ours)
    ours += abs(most_neg_val)
    if max(ours) != min(ours):
        ours /= max(ours) - min(ours)
    assert sum(ours[ours < 0]) == 0, "There are negative values after normalization! Error!"

    # Cross correlate with some amount of zero extension
    xcor = np.correlate(ours, target, mode='full')

    # Find the single maximum value along the xcor vector
    # This is the place at which the waves match each other best
    return max(xcor)


def evaluate(target: str, directory: str) -> None:
    targetseg = asg.from_file(os.path.join(directory, target))
    for i in range(0, 4):
        # Evaluate the cross-correlation between the best agent at that time point and the target
        wavname = "{}.synthmimic_{}.wav".format(target, i)
        wavfpath = os.path.join(directory, wavname)
        assert os.path.isfile(wavfpath), "{} is not a file.".format(wavfpath)

        # Read it in and apply the cross correlation procedure against its target
        seg = asg.from_file(wavfpath)
        assert len(seg) == len(targetseg)
        assert seg.frame_rate == targetseg.frame_rate
        assert seg.sample_width == targetseg.sample_width
        assert seg.channels == targetseg.channels
        xcorval = xcorevaluate(seg, targetseg)
        print(xcorval)

if __name__ == "__main__":
    assert os.path.isdir(baselocdir), "{} is not a directory.".format(baselocdir)

    for target in (target05_1, target05_2):
        for directory in halfsecond_directories:
            evaluate(target, directory)

    # Do the same for 0.3 second condition
    for target in (target03_1, target03_2):
        for directory in thirdsecond_directories:
            evaluate(target, directory)
