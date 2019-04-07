"""
This module contains the external-facing functions for analyzing the
synthesis stuff.
"""
import audiosegment as asg
import os

from experiment.analysis.synthesis import analyze                           # pylint: disable=locally-disabled, import-error


def analyze_pretrained_model(config, resultsdir: str, savetodir: str) -> None:
    """
    Makes the plots and whatever other artifacts are needed for
    analysis of a pretrained articulatory synthesis model.
    """
    # Get needed configuarations
    pngname = config.getstr('experiment', 'name')

    # Find all the sound files in the directory
    soundfpaths = analyze._get_soundfpaths_from_dir(resultsdir)

    # Order the sounds chronologically (in terms of training. So pretraining, then phase1_0, phase1_1, etc.).
    orderedfpaths = analyze._order_fpaths(soundfpaths)

    # Load them all in and resample them to something reasonable
    orderedsegs = [asg.from_file(fp).resample(16000, 2, 1) for fp in orderedfpaths]

    # Save them in the savetodir
    for fpath, seg in zip(orderedfpaths, orderedsegs):
        fname = os.path.basename(fpath)
        savepath = os.path.join(savetodir, fname)
        seg.export(savepath, format='WAV')

    # Plot each one
    analyze._analyze(orderedsegs, pngname, savetodir)
