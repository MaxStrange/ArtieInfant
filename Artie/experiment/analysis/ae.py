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

def _analyze_2d_latent_space(autoencoder: vae.VariationalAutoEncoder, training_root: str, testsplit_root: str, batchsize: int, imshapes: [int], specargs: {}) -> None:
    """
    Analyzes a 2D latent space for an autoencoder.
    """
    analysisdir = os.path.abspath(os.path.dirname(__file__))
    voweldir = os.path.join(analysisdir, "vae", "sounds", "vowels")

    if not isinstance(autoencoder, vae.VariationalAutoEncoder):
        raise NotImplementedError("Currently only able to analyze variational autoencoders. Not sure what will break if not variational.")
    else:
        for directory in (training_root, testsplit_root):
            nworkers = 1
            # Plot the latent space of the encoder
            means, logvars, encodings = plotvae._predict_on_spectrograms(directory, autoencoder, batchsize, nworkers, imshapes)
            stdevs = np.exp(0.5 * logvars)
            plotvae._plot_variational_latent_space(encodings, None, None, means, stdevs, None, None)

            # Plot the latent space of the encoder, but this time with vowels plotted in red
            special_means, special_logvars, special_encodings = plotvae._predict_on_sound_files(None, voweldir, autoencoder, **specargs)
            if special_logvars is not None:
                special_stdevs = np.exp(0.5 * special_logvars)
            plotvae._plot_variational_latent_space(encodings, special_encodings, "vowels", means, stdevs, special_means, special_stdevs)


def analyze_latent_space(autoencoder: vae.VariationalAutoEncoder, nembedding_dims: int, training_root: str, testsplit_root: str, batchsize: int, imshapes: [int], specargs: {}) -> None:
    """
    Analyze the latent space of the given `autoencoder`. This will only work if
    `nembedding_dims` is 1, 2, or 3.
    """
    if nembedding_dims == 1:
        raise NotImplementedError("Currently can't analyze 1-dimensional embeddings. Implement me!")
    elif nembedding_dims == 2:
        _analyze_2d_latent_space(autoencoder, training_root, testsplit_root, batchsize, imshapes, specargs)

        # Plot test split
        # Plot split we trained on
        # Plot both again, but this time with red circles for vowels
    elif nembedding_dims == 3:
        raise NotImplementedError("Currently can't analyze 3-dimensional embeddings. Implement me!")
    else:
        raise ValueError("nembedding_dims must be 1, 2, or 3, but is {}".format(nembedding_dims))

def analyze_reconstruction(autoencoder: vae.VariationalAutoEncoder) -> None:
    """
    Plot an input spectrogram side-by-side with itself after reconstruction
    """
    raise NotImplementedError

def analyze_variational_sampling(autoencoder: vae.VariationalAutoEncoder) -> None:
    """
    If a Variational AE, this samples from latent space and plots a swathe of spectrograms.
    """
    raise NotImplementedError

def analyze(config, autoencoder: vae.VariationalAutoEncoder) -> None:
    """
    Analyzes the given `autoencoder` according to the `config`.
    Saves analysis artifacts in an appropriate place, based again on `config`.
    """
    # Get what we need from the config file
    duration_s      = config.getfloat('preprocessing', 'seconds_per_spectrogram')
    window_length_s = config.getfloat('preprocessing', 'spectrogram_window_length_s')
    overlap         = config.getfloat('preprocessing', 'overlap')
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
        analyze_latent_space(autoencoder, nembedding_dims, training_root, testsplit_root, batchsize, imshapes, specargs)
    else:
        logging.warn("Cannot do any reasonable latent space visualization for embedding spaces of dimensionality greater than 3.")

    analyze_reconstruction(autoencoder)

    if isinstance(autoencoder, vae.VariationalAutoEncoder):
        analyze_variational_sampling(autoencoder)
