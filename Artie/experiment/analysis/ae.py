"""
This is the external interface file for analysis of the autoencoder.
"""
from internals.motorcortex import motorcortex           # pylint: disable=locally-disabled, import-error
from internals.vae import vae                           # pylint: disable=locally-disabled, import-error

import logging

def analyze_latent_space(autoencoder: vae.VariationalAutoEncoder, nembedding_dims: int, training_root: str, testsplit_root: str) -> None:
    """
    Analyze the latent space of the given `autoencoder`. This will only work if
    `nembedding_dims` is 1, 2, or 3.
    """
    if nembedding_dims not in (1, 2, 3):
        raise ValueError("nembedding_dims must be 1, 2, or 3, but is {}".format(nembedding_dims))

    # Plot the latent space for embedding dimensions of 1, 2, or 3 for:
    #       -> test split
    #       -> split we trained on
    # Do the same thing, but this time, add red circles for vowels

    raise NotImplementedError

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
    nembedding_dims = config.getint('autoencoder', 'nembedding_dims')
    testsplit_root  = config.getstr('autoencoder', 'testsplit_root')
    training_root   = config.getstr('autoencoder', 'preprocessed_data_root')

    if nembedding_dims in (1, 2, 3):
        analyze_latent_space(autoencoder, nembedding_dims, training_root, testsplit_root)
    else:
        logging.warn("Cannot do any reasonable latent space visualization for embedding spaces of dimensionality greater than 3.")

    analyze_reconstruction(autoencoder)

    if isinstance(autoencoder, vae.VariationalAutoEncoder):
        analyze_variational_sampling(autoencoder)
