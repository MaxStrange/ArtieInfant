"""
This is the phase 1 file.

This file's API consists simply of the function run(), which will run phase 1 of the thesis experiment.
"""
from experiment.analysis import ae                      # pylint: disable=locally-disabled, import-error
from experiment.analysis import production              # pylint: disable=locally-disabled, import-error
from experiment.configuration import ConfigError        # pylint: disable=locally-disabled, import-error
from internals.motorcortex import motorcortex           # pylint: disable=locally-disabled, import-error
from internals.vae import vae                           # pylint: disable=locally-disabled, import-error
from internals.vae import ae as vanilla                 # pylint: disable=locally-disabled, import-error
from senses.voice_detector import voice_detector as vd  # pylint: disable=locally-disabled, import-error
from senses.dataproviders import sequence as seq        # pylint: disable=locally-disabled, import-error

import audiosegment
import datetime
import imageio
import keras
import logging
import multiprocessing as mp
import numpy as np
import os
import random
import shutil
import sklearn
import tensorflow as tf
import tqdm
if "TRAVIS_CI" not in os.environ:
    import matplotlib.pyplot as plt

# Back-end module
nnbackend = None


def _build_vae1(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds model 1 of the VAE.
    """
    args = (input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop)

    if is_variational:
        autoencoder = nnbackend.build_variational_autoencoder1(*args)
    else:
        autoencoder = nnbackend.build_autoencoder1(*args)

    return autoencoder

def _build_vae2(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds model 2 of the VAE.
    """
    args = (is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop)

    if is_variational:
        autoencoder = nnbackend.build_variational_autoencoder2(*args)
    else:
        autoencoder = nnbackend.build_autoencoder2(*args)

    return autoencoder

def _build_vae(config):
    """
    Builds the Variational AutoEncoder and returns it. Uses parameters from the config file.
    """
    # Get the input shape that the Encoder layer expects
    input_shape = config.getlist('autoencoder', 'input_shape', type=int)

    # Get the dimensionality of the embedding space
    latent_dim = config.getint('autoencoder', 'nembedding_dims')

    # Get the optimizer
    optimizer = config.getstr('autoencoder', 'optimizer')

    # Get the reconstructive loss function
    loss = config.getstr('autoencoder', 'loss')

    # Value between 0 and 1.0 that shows how much of the whole VAE loss function to assign to KL loss
    kl_loss_proportion = config.getfloat('autoencoder', 'kl_loss_proportion')

    # Value between 0 and 1.0 that shows how much of the whole VAE loss function to assign to reconstructive loss
    reconstructive_loss_proportion = config.getfloat('autoencoder', 'reconstructive_loss_proportion')

    # Value between 0 and 1.0 that shows how much of the whole VAE loss function to assign to the variance portion
    std_loss_proportion = config.getfloat('autoencoder', 'std_loss_proportion')

    # Are we variational or vanilla?
    is_variational = config.getbool('autoencoder', 'is_variational')

    # Get TensorBoard directory
    tbdir = config.getstr('autoencoder', 'tbdir')
    assert os.path.isdir(tbdir) or tbdir.lower() == "none", "{} is not a valid directory. Please fix tbdir in 'autoencoder' section of config file.".format(tbdir)
    # Now create a subdirectory in it
    experiment_name = config.getstr('experiment', 'name')
    tbdir = os.path.join(tbdir, experiment_name)
    os.makedirs(tbdir, exist_ok=True)

    # Remove anything in the tbdir already
    if tbdir is not None:
        shutil.rmtree(tbdir)  # Remove the directory and everything in it
        os.mkdir(tbdir)       # Remake the directory

    if list(input_shape) == [241, 20, 1]:
        return _build_vae1(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_proportion, reconstructive_loss_proportion, std_loss_proportion)
    elif list(input_shape) == [81, 18, 1]:
        return _build_vae2(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_proportion, reconstructive_loss_proportion, std_loss_proportion)
    else:
        raise ValueError("Spectrogram shape must be one of the allowed input shapes for the different VAE models, but is {}".format(input_shape))

def _train_vae(autoencoder, config):
    """
    Train the given `autoencoder` according to parameters listed in `config`.
    """
    # The root of the preprocessed data directory
    root = config.getstr('autoencoder', 'preprocessed_data_root')
    assert os.path.isdir(root), "{} is not a valid path.".format(root)

    # The root of the test split
    test = config.getstr('autoencoder', 'testsplit_root')
    if test.endswith("useless_subdirectory") or test.endswith("useless_subdirectory" + os.sep):
        test = os.path.dirname(test)
    assert os.path.isdir(test), "{} is not a valid path.".format(test)

    # Get whether or not we should visualize during training
    visualize = config.getbool('autoencoder', 'visualize')

    # The number of spectrograms per batch
    batchsize = config.getint('autoencoder', 'batchsize')

    # The number of workers to help with the data collection and feeding
    nworkers = config.getint('autoencoder', 'nworkers')

    # The number of epochs to train. Note that we define how long an epoch is manually
    nepochs = config.getint('autoencoder', 'nepochs')

    # The number of steps we have in an epoch
    steps_per_epoch = config.getint('autoencoder', 'steps_per_epoch')

    # The shape of the images
    imshapes = config.getlist('autoencoder', 'input_shape')[0:2]  # take only the first two dimensions (not channels)
    imshapes = [int(i) for i in imshapes]

    nnbackend.train(autoencoder, visualize, root, steps_per_epoch, imshapes, batchsize, nworkers, test, nepochs)

def _infer_with_vae(autoencoder: vae.VariationalAutoEncoder, config) -> [(str, str, np.array)]:
    """
    Returns a list of tuples of the form (spectrogram_fpath, audiofile_fpath, embedding coordinates as NP array).
    """
    if autoencoder is None:
        raise NotImplementedError("Currently, we need an autoencoder here.")

    try:
        # Try interpreting this configuration as an int - if so, that's the number of random
        # targets drawn from the test split
        print("Finding all the images in the test split...")
        testdir = config.getstr('autoencoder', 'testsplit_root')
        pathnames = [p for p in os.listdir(testdir) if os.path.splitext(p)[1].lower() == ".png"]
        paths = [os.path.abspath(os.path.join(testdir, p)) for p in pathnames]

        # Draw n random items from the test split as our targets
        nitems_to_mimic = config.getint('synthesizer', 'mimicry-targets')
        mimicry_targets = [random.choice(paths) for _ in range(nitems_to_mimic)]
    except ConfigError:
        # Other possibility is that the configuration item is a list of file paths to target
        mimicry_targets = config.getlist('synthesizer', 'mimicry-targets')

    print("Reading in all the images...")
    specs = [imageio.imread(p) / 255.0 for p in mimicry_targets]
    specs = np.expand_dims(np.array(specs), -1)

    logging.info("Found {} spectrograms to feed into the autoencoder at inference time.".format(specs.shape[0]))

    print("Predicting on each image in the mimicry list...")
    if isinstance(autoencoder, vae.VariationalAutoEncoder):
        # The output of the encoder portion of the model is three items: Mean, LogVariance, and Value sampled from described distribution
        # We will only use the means of the distros, not the actual encodings, as the means are simply
        # what is approximated by the encodings (though with uncertainty - the amount of uncertainty is measured in _logvars)
        means, _logvars, _encodings = autoencoder._encoder.predict(specs)
    else:
        # The encoder only outputs embeddings. But these are functionally the same as means for our purposes.
        means = autoencoder._encoder.predict(specs)

    # We have a list of .png files. But we want the WAVs that they correspond to. Let's find them.
    pngs_and_means = [tup for tup in zip(mimicry_targets, means)]
    folder = config.getstr('preprocessing', 'folder_to_save_wavs')  # This is where we saved the corresponding wav files
    fpaths_and_means = [(p, ae.convert_spectpath_to_audiofpath(folder, p), embedding) for p, embedding in pngs_and_means]
    if len(fpaths_and_means) < 10:
        logging.info("mimicry targets:\n{}".format(fpaths_and_means))
    else:
        logging.info("Too many mimicry targets to list. Have: {}".format(len(fpaths_and_means)))

    return fpaths_and_means

def _train_or_load_autoencoder(train_vae: bool, config) -> vae.VariationalAutoEncoder:
    """
    If `train_vae` is `True`, we use the config file to determine stuff and train a VAE.
    If it is `False`, we attempt to use the config to load an already trained VAE.
    If we cannot find one, we return None and issue a warning over logging.
    """
    # Build the right autoencoder model
    autoencoder = _build_vae(config)

    if train_vae:
        name = config.getstr('experiment', 'name')
        ae_savedir = config.getstr('autoencoder', 'weights_path')
        ae_savefpath = os.path.join(ae_savedir, name)
        fpath_to_save = "{}.h5".format(ae_savefpath)

        # Train the autoencoder
        logging.info("Training the autoencoder. Models will be saved to: {}".format(fpath_to_save))
        _train_vae(autoencoder, config)

        # Save the model's weights
        autoencoder.save_weights(fpath_to_save)
    else:
        try:
            # Load the weights into the constructed autoencoder model
            autoencoder_weights_fpath = config.getstr('autoencoder', 'weights_path')
            logging.info("Attempting to load autoencoder weights from {}".format(autoencoder_weights_fpath))
            autoencoder.load_weights(autoencoder_weights_fpath)
        except OSError:
            # Couldn't find a model and we weren't told to train one. Hopefully user knows what they're doing
            logging.warn("Could not find any autoencoder.")
            autoencoder = None

    return autoencoder

def run(config, savedir, preprocess=False, preprocess_part_two=False, pretrain_synth=False, train_vae=False, train_synth=False, network_backend=None):
    """
    Entry point for Phase 1.

    Initializes and pretrains the voice synthesization network to vocalize;
    Creates and trains a variational autoencoder on the entire Oliver data set;
    Applies a clustering algorithm to the embeddings that the VAE comes up with once it is trained;
    Determines a prototype sound for each cluster;
    Finishes training the voice synthesizer to mimic these sounds based on which embedding it observes.

    If `preprocess` is True, we will preprocess all the data as part of the experiment. See the config file for details.
    If `preprocess_part_two` is True, we will convert all the preprocessed sound files into black and white images of spectrograms.
    If `pretrain_synth` is True, we will pretrain the voice synthesizer to make noise.
    If `train_vae` is True, we will train the variational autoencoder on the preprocessed data.
    If `train_synth` is True, we will train the voice synthesizer to mimic the prototypical proto phonemes.

    `network_backend` specifies which deep learning back-end to use and must be a module found in the backend folder.
    """
    global nnbackend
    nnbackend = nnbackend

    # Potentially preprocess the audio
    if preprocess:
        print("Preprocessing all sounds. This will take close to forever...")
        _run_preprocessing_pipeline(config)

    # Convert the preprocessed audio files into spectrograms and save them as image files
    if preprocess_part_two:
        print("Converting all preprocessed sound files into spectrograms and saving them as images. This will take the rest of forever...")
        _convert_to_images(config)

    # Pretrain the voice synthesizer to make non-specific noise
    if pretrain_synth:
        print("Pretraining the voice synthesizer. Learning to coo...")
        synthmodel = motorcortex.SynthModel(config)
        synthmodel.pretrain()
        production.analyze_pretrained_model(config, synthmodel.phase0_artifacts_dir, savedir, "pretraining", synthmodel)
    else:
        synthmodel = None

    # Get a trained autoencoder
    autoencoder = _train_or_load_autoencoder(train_vae, config)
    if train_vae:
        print("Analyzing the autoencoder...")
        ae.analyze(config, autoencoder, savedir)

    # Train the motor cortex to produce sounds from these different embeddings
    # The synthesizer uses the autoencoder to evaluate how close its output is to the target sound
    # in latent space.
    if train_synth:
        # Now use the VAE on the test split and save pairs of (audiofile, coordinates in embedding space)
        mimicry_targets = _infer_with_vae(autoencoder, config)

        if synthmodel is None:
            synthmodel = motorcortex.SynthModel(config)

        trained_synth_models = motorcortex.train_on_targets(config, synthmodel, mimicry_targets, autoencoder)
        production.analyze_models(config, trained_synth_models, savedir, [target[1] for target in mimicry_targets])
