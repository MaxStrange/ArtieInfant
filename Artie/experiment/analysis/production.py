"""
This module contains the external-facing functions for analyzing the
synthesis stuff.
"""
import audiosegment as asg
import os
import pickle

from experiment.analysis.synthesis import analyze                           # pylint: disable=locally-disabled, import-error
from internals.motorcortex import motorcortex                               # pylint: disable=locally-disabled, import-error

def analyze_pretrained_model(config, resultsdir: str, savetodir: str, targetname: str, model: motorcortex.SynthModel) -> None:
    """
    Makes the plots and whatever other artifacts are needed for
    analysis of a pretrained articulatory synthesis model.
    """
    # Get needed configuarations
    pngname = config.getstr('experiment', 'name')
    window_length_s = config.getfloat('preprocessing', 'spectrogram_window_length_s')
    overlap = config.getfloat('preprocessing', 'spectrogram_window_overlap')
    sample_rate_hz = config.getfloat('preprocessing', 'spectrogram_sample_rate_hz')

    pngname += "_" + targetname

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
    analyze._analyze(orderedsegs, pngname, savetodir, window_length_s, overlap, sample_rate_hz)

    # Save the model as well
    savepath = os.path.join(savetodir, targetname + ".pkl")
    print("Saving", savepath)
    model.save(savepath)

def analyze_models(config, trained_models: [motorcortex.SynthModel], savetodir: str) -> None:
    """
    Similar to `analyze_pretrained_model`, but for a list of trained models.
    """
    # Currently, we just do the same thing as the pretrained analysis...
    for model in trained_models:
        analyze_pretrained_model(config, model.phase1_artifacts_dir, savetodir, os.path.basename(model.target), model)
