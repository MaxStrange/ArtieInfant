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
import os

def _get_soundfpaths_from_dir(resultsdir):
    """
    Returns a list of fpaths corresponding to all the OGG or WAV files
    in the directory (non-recursively).
    """
    fpaths = []
    for fname in os.listdir(resultsdir):
        if os.path.splitext(fname)[-1].lower() in (".ogg", ".wav"):
            fpaths.append(os.path.join(resultsdir, fname))
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

def _maximize():
    """
    Maximizes the matplotlib window.
    """
    manager = plt.get_current_fig_manager()

    backend = plt.get_backend()
    if backend.lower().startswith("qt"):
        manager.window.showMaximized()
    elif backend.lower().startswith("tk"):
        manager.resize(*manager.window.maxsize())
    elif backend.lower().startswith("wx"):
        manager.frame.Maximize(True)

def _analyze(segments, targetname, savetodir, window_length_s, overlap, sample_rate_hz):
    """
    Plots and saves figures.
    """
    segments = [s.resample(sample_rate_hz) for s in segments]

    # Plot each wave form
    for i, s in enumerate(segments):
        plt.subplot(len(segments), 1, i + 1)
        plt.plot(s.to_numpy_array())
    _maximize()
    plt.title("Waveform")
    plt.ylabel("PCM")
    plt.xlabel("Sample")
    save = os.path.join(savetodir, "{}.png".format(targetname))
    print("Saving", save)
    plt.savefig(save)
    plt.clf()

    # Now plot each spectrogram
    for i, s in enumerate(segments):
        plt.subplot(len(segments), 1, i + 1)
        fs, ts, amps = s.spectrogram(window_length_s=window_length_s, overlap=overlap, window=('tukey', 0.5))
        plt.pcolormesh(ts, fs, amps)
    plt.title("Spectrogram")
    save = os.path.join(savetodir, "{}_spectrogram.png".format(targetname))
    print("Saving", save)
    plt.savefig(save)
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("resultsdir", help="A directory of results from the articulatory synthesis model.")
    args = parser.parse_args()

    if not os.path.isdir(args.resultsdir):
        print("{} is not a valid directory.".format(args.resultsdir))
        exit(1)

    # Find all the sound files in the directory
    soundfpaths = _get_soundfpaths_from_dir(args.resultsdir)

    # Order the sounds chronologically (in terms of training. So pretraining, then phase1_0, phase1_1, etc.).
    orderedfpaths = _order_fpaths(soundfpaths)

    # Load them all in and resample them appropriately
    orderedsegs = [asg.from_file(fp).resample(16000, 2, 1) for fp in orderedfpaths]

    # Plot each one
    _analyze(orderedsegs, os.path.basename(args.resultsdir.strip(os.sep)), ".", 0.5, 0.2, 16000.0)
