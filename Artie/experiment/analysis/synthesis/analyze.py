"""
Use this script to analyze articulatory synthesis results.

Expects a directory of structure:

```
results
|-target.wav/ogg
|-<Phase0OutputSound.wav>
|-Phase1Output_<x>.wav
```
"""
import argparse
import audiosegment as asg
import matplotlib.pyplot as plt
import numpy as np
import os

def _get_soundfpaths_from_dir(resultsdir, targetname=None):
    """
    Returns a list of fpaths corresponding to all the OGG or WAV files
    in the directory (non-recursively) that have `target` in their name.

    If `target` is None, we return all OGG and WAV files in the directory.
    """
    print("Looking for files with '{}' in them...".format(targetname))
    fpaths = []
    for fname in os.listdir(resultsdir):
        # Are we interested in this file?
        is_ogg_or_wav = os.path.splitext(fname)[-1].lower() in (".ogg", ".wav")
        has_targetname = targetname is None or targetname in fname

        if is_ogg_or_wav and has_targetname:
            fpaths.append(os.path.join(resultsdir, fname))

    print("Found: {}".format(fpaths))
    return fpaths

def _order_fpaths(soundfpaths):
    """
    Order the sound files according to their phase and articulation group.
    So, Phase0 comes first, then Phase1_0, then Phase1_1, etc.
    If there are any left over, they are appended at the end.
    """
    ordered = []
    soundfpaths = list(soundfpaths)  # We will be modifying this list

    # Find all the Phase 1 items and put them in in any order
    for fpath in soundfpaths:
        if os.path.basename(fpath).lower().startswith("phase1output_"):
            ordered.append(fpath)

    # Remove from soundfpaths
    for fpath in ordered:
        soundfpaths.remove(fpath)

    # Sort according to their indexes (file names look like Phase1Output_X.wav),
    # where X is an integer. Extract that integer and sort according to it.
    ordered = sorted(ordered, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Find if there is a Phase 0 and put it in first
    for fpath in soundfpaths:
        if os.path.basename(fpath).lower() == "phase0outputsound.wav":
            ordered.insert(0, fpath)
            soundfpaths.remove(fpath)
            break

    # Any remaining items go in last in whatever order
    for fpath in soundfpaths:
        ordered.append(fpath)

    return ordered

def _analyze(segments, targetname, savetodir, window_length_s, overlap, sample_rate_hz):
    """
    Plots and saves figures.
    """
    segments = [s.resample(sample_rate_hz) for s in segments]

    # Plot each wave form
    fig, axs = plt.subplots(len(segments), 1, constrained_layout=False, squeeze=False)
    for i, s in enumerate(segments):
        arr = s.to_numpy_array()
        times = np.linspace(0, len(arr) / s.frame_rate, num=len(arr))
        axs[i][0].plot(times, arr)
        if i == len(segments) - 1:
            # This is the bottom row, enable the xticks
            pass
        else:
            # Disable the xticks
            axs[i][0].xaxis.set_ticklabels([])
    axs[-1][0].set_xlabel("Time (s)")
    fig.suptitle("Waveforms of Generated Utterances")
    fig.text(0.01, 0.5, "PCM", ha='center', va='center', rotation='vertical')

    # Save the plot
    save = os.path.join(savetodir, "{}.png".format(targetname))
    print("Saving", save)
    plt.savefig(save)
    plt.clf()

    # Now plot each spectrogram
    fig, axs = plt.subplots(len(segments), 1, constrained_layout=False, squeeze=False)
    for i, s in enumerate(segments):
        fs, ts, amps = s.spectrogram(window_length_s=window_length_s, overlap=overlap, window=('tukey', 0.5))
        axs[i][0].pcolormesh(ts, fs, amps)
        if i == len(segments) - 1:
            # This is the bottom row, enable the xticks
            pass
        else:
            # Disable the xticks
            axs[i][0].xaxis.set_ticklabels([])
    axs[-1][0].set_xlabel("Time (s)")
    fig.suptitle("Spectrograms of Generated Utterances")
    fig.text(0.03, 0.5, "Hz", ha='center', va='center', rotation='vertical')

    # Save the plot
    save = os.path.join(savetodir, "{}_spectrogram.png".format(targetname))
    print("Saving", save)
    plt.savefig(save)
    plt.clf()
