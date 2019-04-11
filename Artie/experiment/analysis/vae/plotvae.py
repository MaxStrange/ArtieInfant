"""
Load the given spectrogram model, run a bunch of spectrograms through it,
then see what it does with them in its 2D latent space.
"""
from mpl_toolkits.mplot3d import Axes3D
import argparse
import audiosegment as asg
import imageio
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
from keras import preprocessing

from experiment.thesis import phase1 as p1                      # pylint: disable=locally-disabled, import-error
import experiment.configuration as configuration                # pylint: disable=locally-disabled, import-error
import internals.vae.vae as vae                                 # pylint: disable=locally-disabled, import-error

def load_spectrograms_from_directory(d, numspecs=None):
    """
    Loads up to numspecs spectrograms from d. If numspecs is None, we load all of them.
    Looks for spectrograms non-recursively.

    Returns them as a numpy array of shape (numspecs, freqs, times, colorchannels). See the config file.
    """
    assert os.path.isdir(d), "'d' must be a valid directory, but was passed {}".format(d)
    assert numspecs is None or numspecs >= 0, "'numspecs' must be either None or a positive number"
    if numspecs is not None:
        numspecs = int(round(numspecs))

    pathnames = [p for p in os.listdir(d) if os.path.splitext(p)[1].lower() == ".png"]
    if not pathnames:
        # We couldn't find any files in the supplied directory. Sometimes I am stupid though,
        # so let's try whatever/useless_subdirectory as well.
        new_d = os.path.join(d, "useless_subdirectory")
        pathnames = [p for p in os.listdir(new_d) if os.path.splitext(p)[1].lower() == ".png"]
        if pathnames:
            # It worked, so let's update d to be this new one that actually has stuff in it
            d = new_d

    if numspecs:
        pathnames = pathnames[:numspecs]

    if not pathnames:
        raise FileNotFoundError("Could not find any .png files in directory {}".format(d))

    paths = [os.path.abspath(os.path.join(d, p)) for p in pathnames]
    specs = [imageio.imread(p) / 255.0 for p in paths]
    arr = np.array(specs)
    return np.expand_dims(arr, -1)

def analyze_single_segment(model: vae.VariationalAutoEncoder, targetfpath: str, visualize=False):
    """
    Plot the location of the given segment after we encode it. Plots the average of 100 encodings
    of it as well as the average distribution from the 100 tries.
    Also plots the 100 encodings and distributions.

    If visualize is True, we will plot stuff.
    """
    sample_rate_hz  = 16000.0    # 16kHz sample rate
    bytewidth       = 2          # 16-bit samples
    nchannels       = 1          # mono
    duration_s      = 0.5        # Duration of each complete spectrogram
    window_length_s = 0.03       # How long each FFT is
    overlap         = 0.2        # How much each FFT overlaps with each other one

    # Load the audio file into an AudioSegment
    seg = asg.from_file(targetfpath)
    seg = seg.resample(sample_rate_Hz=sample_rate_hz, sample_width=bytewidth, channels=nchannels)

    start_s = 0
    _frequencies, _times, amplitudes = seg.spectrogram(start_s, duration_s, window_length_s=window_length_s, overlap=overlap, window=('tukey', 0.5))
    amplitudes *= 255.0 / np.max(np.abs(amplitudes))
    amplitudes = amplitudes / 255.0
    amplitudes = np.expand_dims(amplitudes, -1)  # add color channel
    amplitudes = np.repeat(amplitudes[np.newaxis, :, :, :], 100, axis=0)  # Repeat into batch dimension
    means, logvars, encodings = model._encoder.predict(amplitudes, batch_size=None, steps=1)

    stdevs = np.exp(0.5 * logvars)

    if visualize:
        # Plot where each encoding is
        title = "Scatter Plot of Encodings for {}".format(targetfpath)
        plt.scatter(encodings[:, 0], encodings[:, 1])
        plt.title(title)
        print("Saving", title)
        plt.savefig(title)
        plt.clf()

        # Plot the distributions as circles whose means determine location and whose radii are composed
        # of the standard deviations
        plt.scatter(means[:, 0], means[:, 1], s=np.square(stdevs * 10))
        title = "Distributions the Encodings were drawn From for {}".format(targetfpath)
        plt.title(title)
        print("Saving", title)
        plt.savefig(title)
        plt.clf()

    embedding = np.mean(encodings, axis=0)
    mean = np.mean(means, axis=0)
    stdev = np.mean(stdevs, axis=0)
    return embedding, mean, stdev

def _validate_args(args):
    """
    Validates the arguments and exits if any of them do not make sense.
    """
    if not os.path.isfile(args.model):
        print("{} is not a valid path to a VAE model.".format(args.model))
        exit(1)
    if not os.path.isdir(args.specdir):
        print("{} is not a valid path to a directory of spectrograms.".format(args.specdir))
        exit(2)
    if args.file:
        for fpath in args.file:
            if not os.path.isfile(fpath):
                print("{} is not a valid path to a sound file.".format(fpath))
                exit(3)
    if args.dir and not os.path.isdir(args.dir):
        print("{} is not a valid directory of sound files.".format(args.dir))
        exit(4)

def _build_the_vae(config, model):
    autoencoder = p1._build_vae(config)
    try:
        autoencoder.load_weights(model)
    except Exception as e:
        print("Something went wrong while trying to load the given model. Perhaps the weights don't match with the current architecture? {}".format(e))
        exit(5)
    return autoencoder

def _predict_on_spectrograms(specdir: str, autoencoder: vae.VariationalAutoEncoder, batchsize: int, nworkers: int, imshapes: [int]):
    """
    Returns the values of the spectrogram predictions for each spectrogram found in specdir.
    """
    # Load a bunch of spectrograms into a batch
    specs = load_spectrograms_from_directory(specdir)
    nspecs = specs.shape[0]
    if nspecs == 0:
        logging.warn("Could not find any spectrograms in {}. Trying {}/useless_subdirectory as well.".format(specdir, specdir))
        specs = load_spectrograms_from_directory(os.path.join(specdir, "useless_subdirectory"))
        nspecs = specs.shape[0]
        if nspecs == 0:
            print("Could not find any spectrograms in {} or {}/useless_subdirectory. Cannot predict using them.")
            return
    try:
        aeoutput = autoencoder._encoder.predict(specs)
    except Exception:
        print("Probably out of memory. Trying as an image generator instead.")
        specs = None  # Hint to the GC

        # Remove the useless subdirectory from the path (the imagedatagen needs it, but can't be told about it... ugh)
        pathsplit = specdir.rstrip(os.sep).split(os.sep)
        root = os.path.join(*[os.sep if p == '' else p for p in pathsplit[0:-1]])

        # Set up based on config file
        imreader = preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
        print("Creating datagen...")
        datagen = imreader.flow_from_directory(root,
                                            target_size=imshapes,
                                            color_mode='grayscale',
                                            classes=None,
                                            class_mode='input',
                                            batch_size=batchsize,
                                            shuffle=True)
        print("Predicting...")
        aeoutput = autoencoder._encoder.predict_generator(datagen,
                                            steps=int(nspecs / batchsize),
                                            use_multiprocessing=False,
                                            workers=nworkers)
    if isinstance(autoencoder, vae.VariationalAutoEncoder):
        means, logvars, encodings = aeoutput
    else:
        means = None
        logvars = None
        encodings = aeoutput

    return means, logvars, encodings

def _predict_on_sound_files(fpaths: [str], dpath: str, model: vae.VariationalAutoEncoder,
    sample_rate_hz=16000.0, bytewidth=2, nchannels=1, duration_s=0.5, window_length_s=0.03, overlap=0.2):
    """
    Run the given model on each file in fpaths and each file in dpath. These are sound files, not spectrograms,
    so they need to be converted to spectrograms first.

    If fpaths and dpaths are both None or empty, we return None, None, None.
    """
    if dpath is None:
        dpath = []
    if fpaths is None:
        fpaths = []

    # Combine all the fpaths into a single fpath list
    if dpath:
        for root, _dnames, fnames in os.walk(dpath):
            for fname in fnames:
                fpath = os.path.join(root, fname)
                fpaths.append(fpath)

    # Load all the segments and resample them appropriately
    segs = []
    for fpath in fpaths:
        try:
            segs.append(asg.from_file(fpath).resample(sample_rate_hz, bytewidth, nchannels))
        except asg.pydub.audio_segment.CouldntDecodeError:
            print("NOTE: Couldn't decode {} as an audio file.".format(fpath))
            continue

    # Convert each segment into a spectrogram
    specs = []
    for seg in segs:
        start_s = 0
        _frequencies, _times, amplitudes = seg.spectrogram(start_s, duration_s, window_length_s=window_length_s, overlap=overlap, window=('tukey', 0.5))
        amplitudes *= 255.0 / np.max(np.abs(amplitudes))
        amplitudes = amplitudes / 255.0
        amplitudes = np.expand_dims(amplitudes, -1)  # add color channel
        specs.append(amplitudes)
    specs = np.array(specs)

    # Predict from the encoder portion of the model
    if specs.shape[0] > 0:
        modelret = model._encoder.predict(specs)
        if isinstance(model, vae.VariationalAutoEncoder):
            means, logvars, encodings = modelret
        else:
            means = None
            logvars = None
            encodings = modelret
    else:
        means, logvars, encodings = None, None, None
    return means, logvars, encodings

def _plot_vanilla_latent_space(encodings, special_encodings, name, savedir, *, ndims=2):
    """
    See `_plot_variational_latent_space`.
    """
    # Plot where each embedding is
    if ndims == 1:
        plt.title("Scatter Plot of Embeddings")
        plt.scatter(encodings, np.zeros_like(encodings))
        if special_encodings is not None:
            plt.scatter(special_encodings, np.zeros_like(special_encodings), c='red')
    elif ndims == 2:
        plt.title("Scatter Plot of Embeddings")
        plt.scatter(encodings[:, 0], encodings[:, 1])
        if special_encodings is not None:
            plt.scatter(special_encodings[:, 0], special_encodings[:, 1], c='red')
    elif ndims == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(encodings[:, 0], encodings[:, 1], encodings[:, 2])
        if special_encodings is not None:
            ax.scatter(special_encodings[:, 0], special_encodings[:, 1], special_encodings[:, 2], c='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Scatter Plot of Embeddings")
    else:
        raise ValueError("`ndims` must be 1, 2, or 3, but is {}".format(ndims))

    save = os.path.join(savedir, "scatter_{}_embeddings_{}.png".format(encodings.shape[0], name))
    print("Saving", save)
    plt.savefig(save)
    plt.clf()

def _plot_variational_latent_space(encodings, special_encodings, name, means, stdevs, special_means, special_stdevs, savedir, *, ndims=2):
    """
    Does the plotting that all the rest of this file's functions are centered around.

    :param encodings: The embeddings that we will plot in blue.
    :param special_encodings: Embeddings to plot in red.
    :param name: Name of the group of the special encodings.
    :param means: Means of the blues.
    :param stdevs: STDevs of the blues.
    :param special_means: Means of the reds.
    :param special_stdevs: STDevs of the reds.
    :param savedir: The directory to save the artifacts to.
    """
    _plot_vanilla_latent_space(encodings, special_encodings, name, savedir, ndims=ndims)

    # Plot the distributions as circles whose means determine location and whose radii are composed
    # of the standard deviations
    if ndims == 1:
        plt.title("Distributions the Embeddings were drawn From")
        plt.scatter(means, np.zeros_like(means), s=np.square(stdevs * 10))
        if special_means is not None:
            plt.scatter(special_means, np.zeros_like(special_means), s=np.square(special_stdevs * 10), c='red')
    elif ndims == 2:
        plt.title("Distributions the Embeddings were drawn From")
        plt.scatter(means[:, 0], means[:, 1], s=np.square(stdevs * 10))
        if special_means is not None:
            plt.scatter(special_means[:, 0], special_means[:, 1], s=np.square(special_stdevs * 10), c='red')
    elif ndims == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(means[:, 0], means[:, 1], means[:, 2], s=np.square(stdevs * 10))
        if special_encodings is not None:
            ax.scatter(special_means[:, 0], special_means[:, 1], special_means[:, 2], s=np.square(special_stdevs * 10), c='red')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Distributions the Embeddings were drawn From")
    else:
        raise ValueError("`ndims` must be 1, 2, or 3, but is {}".format(ndims))

    save = os.path.join(savedir, "scatter_{}_distros_{}.png".format(encodings.shape[0], name))
    print("Saving", save)
    plt.savefig(save)
    plt.clf()
