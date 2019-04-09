"""
This is the external interface file for analysis of the autoencoder.
"""
from internals.motorcortex import motorcortex           # pylint: disable=locally-disabled, import-error
from internals.vae import vae                           # pylint: disable=locally-disabled, import-error
from experiment.analysis.vae import plotvae             # pylint: disable=locally-disabled, import-error
from experiment.analysis.vae import testvae             # pylint: disable=locally-disabled, import-error

import logging
import numpy as np
import os

def _analyze_latent_space(autoencoder: vae.VariationalAutoEncoder, training_root: str, testsplit_root: str, batchsize: int, imshapes: [int], specargs: {}, savedir: str, ndims: int) -> None:
    """
    Analyzes a 2D latent space for an autoencoder.
    """
    analysisdir = os.path.abspath(os.path.dirname(__file__))
    voweldir = os.path.join(analysisdir, "vae", "sounds", "vowels")

    for directory in (training_root, testsplit_root):
        nworkers = 1
        # Plot the latent space of the encoder
        print("Visualizing latent space for {}...".format(directory))
        means, logvars, encodings = plotvae._predict_on_spectrograms(directory, autoencoder, batchsize, nworkers, imshapes)
        if isinstance(autoencoder, vae.VariationalAutoEncoder):
            stdevs = np.exp(0.5 * logvars)
            plotvae._plot_variational_latent_space(encodings, None, None, means, stdevs, None, None, savedir, ndims=ndims)
        else:
            plotvae._plot_vanilla_latent_space(encodings, None, None, savedir, ndims=ndims)

        # Plot the latent space of the encoder, but this time with vowels plotted in red
        print("Visualizing vowels in the latent space...")
        special_means, special_logvars, special_encodings = plotvae._predict_on_sound_files(None, voweldir, autoencoder, **specargs)
        if isinstance(autoencoder, vae.VariationalAutoEncoder):
            if special_logvars is not None:
                special_stdevs = np.exp(0.5 * special_logvars)
            plotvae._plot_variational_latent_space(encodings, special_encodings, "vowels", means, stdevs, special_means, special_stdevs, savedir, ndims=ndims)
        else:
            plotvae._plot_vanilla_latent_space(encodings, special_encodings, "vowels", savedir, ndims=ndims)

def analyze_latent_space(autoencoder: vae.VariationalAutoEncoder, nembedding_dims: int, training_root: str, testsplit_root: str, batchsize: int, imshapes: [int], specargs: {}, savedir: str) -> None:
    """
    Analyze the latent space of the given `autoencoder`. This will only work if
    `nembedding_dims` is 1, 2, or 3.
    """
    if nembedding_dims in (1, 2, 3):
        _analyze_latent_space(autoencoder, training_root, testsplit_root, batchsize, imshapes, specargs, savedir, nembedding_dims)
    else:
        raise ValueError("nembedding_dims must be 1, 2, or 3, but is {}".format(nembedding_dims))

def analyze_reconstruction(audiofpaths, impaths, autoencoder: vae.VariationalAutoEncoder, savedir: str) -> None:
    """
    Plot an input spectrogram side-by-side with itself after reconstruction
    """
    for audiofpath, impath in zip(audiofpaths, impaths):
        testvae._plot_input_output_spectrograms(audiofpath, impath, autoencoder, savedir)

def analyze_variational_sampling(autoencoder: vae.VariationalAutoEncoder, shape: [int], low: float, high: float, savedir: str, ndims: int) -> None:
    """
    If a Variational AE, this samples from latent space and plots a swathe of spectrograms.
    """
    testvae._plot_samples_from_latent_space(autoencoder, shape, savedir, ndims)
    if ndims < 3:
        testvae._plot_topographic_swathe(autoencoder, shape, low, high, savedir, ndims)

def convert_spectpath_to_audiofpath(audiofolder: str, specpath: str) -> str:
    """
    Finds the path of the audio file that corresponds to the spectrogram
    found at `specpath`.
    """
    specfname = os.path.basename(specpath)
    wavfname = os.path.splitext(specfname)[0] + ".wav"
    wavfpath = os.path.join(audiofolder, wavfname)
    if not os.path.isfile(wavfpath):
        raise FileNotFoundError("Could not find {}.".format(wavfpath))
    return wavfpath

def analyze(config, autoencoder: vae.VariationalAutoEncoder, savedir: str) -> None:
    """
    Analyzes the given `autoencoder` according to the `config`.
    Saves analysis artifacts in an appropriate place, based again on `config`.
    """
    # Get what we need from the config file
    swathe_low      = config.getfloat('autoencoder', 'topographic_swathe_low')
    swathe_high     = config.getfloat('autoencoder', 'topographic_swathe_high')
    reconspects     = config.getlist('autoencoder', 'spectrograms_to_reconstruct')
    audiofolder     = config.getstr('preprocessing', 'folder_to_save_wavs')  # This is where we saved the corresponding wav files
    duration_s      = config.getfloat('preprocessing', 'seconds_per_spectrogram')
    window_length_s = config.getfloat('preprocessing', 'spectrogram_window_length_s')
    overlap         = config.getfloat('preprocessing', 'spectrogram_window_overlap')
    sample_rate_hz  = config.getfloat('preprocessing', 'spectrogram_sample_rate_hz')
    bytewidth       = config.getint('preprocessing', 'bytewidth')
    nembedding_dims = config.getint('autoencoder', 'nembedding_dims')
    testsplit_root  = config.getstr('autoencoder', 'testsplit_root')
    training_root   = config.getstr('autoencoder', 'preprocessed_data_root')
    batchsize       = config.getint('autoencoder', 'batchsize')
    imshapes        = config.getlist('autoencoder', 'input_shape')[0:2]  # take only the first two dimensions (not channels)
    imshapes        = [int(i) for i in imshapes]  # They are strings in the config file, so convert them to ints
    nchannels       = 1

    # These are special because I'm lazy. I'll leave it at that...
    specargs = {
        'sample_rate_hz': sample_rate_hz,
        'bytewidth': bytewidth,
        'nchannels': nchannels,
        'duration_s': duration_s,
        'window_length_s': window_length_s,
        'overlap': overlap,
    }

    if nembedding_dims in (1, 2, 3):
        analyze_latent_space(autoencoder, nembedding_dims, training_root, testsplit_root, batchsize, imshapes, specargs, savedir)
    else:
        logging.warn("Cannot do any reasonable latent space visualization for embedding spaces of dimensionality greater than 3.")

    print("Analyzing reconstruction...")
    # Get all the audio files that correspond to the reconstruction spectrograms
    reconaudiofpaths = [convert_spectpath_to_audiofpath(audiofolder, p) for p in reconspects]
    analyze_reconstruction(reconaudiofpaths, reconspects, autoencoder, savedir)

    if isinstance(autoencoder, vae.VariationalAutoEncoder):
        print("Analyzing variational stuff...")
        analyze_variational_sampling(autoencoder, imshapes, swathe_low, swathe_high, savedir, nembedding_dims)
