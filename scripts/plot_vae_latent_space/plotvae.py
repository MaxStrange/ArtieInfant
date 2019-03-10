"""
Load the given spectrogram model, run a bunch of spectrograms through it,
then see what it does with them in its 2D latent space.
"""
import audiosegment as asg
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
from keras import preprocessing

sys.path.append(os.path.abspath("../../Artie/experiment"))
sys.path.append(os.path.abspath("../../Artie"))
import thesis.phase1 as p1                                      # pylint: disable=locally-disabled, import-error
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
    paths = [os.path.abspath(os.path.join(d, p)) for p in pathnames]
    specs = [imageio.imread(p) / 255.0 for p in paths]
    arr = np.array(specs)
    return np.expand_dims(arr, -1)

def analyze_single_segment(model: vae.VariationalAutoEncoder, targetfpath: str):
    """
    Plot the location of the given segment after we encode it. Plots the average of 100 encodings
    of it as well as the average distribution from the 100 tries.
    Also plots the 100 encodings and distributions.
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
    amplitudes = amplitudes.astype(np.uint8)
    amplitudes = np.expand_dims(amplitudes, -1)  # add color channel
    amplitudes = np.repeat(amplitudes[np.newaxis, :, :, :], 100, axis=0)  # Repeat into batch dimension
    means, logvars, encodings = model._encoder.predict(amplitudes, batch_size=None, steps=1)

    stdevs = np.exp(0.5 * logvars)
    # Visualize where the encodings ended up

    # Plot where each encoding is
    plt.scatter(encodings[:, 0], encodings[:, 1])
    plt.title("Scatter Plot of Encodings for {}".format(targetfpath))
    plt.show()

    # Plot the distributions as circles whose means determine location and whose radii are composed
    # of the standard deviations
    plt.scatter(means[:, 0], means[:, 1], s=np.square(stdevs * 10))
    plt.title("Distributions the Encodings were drawn From for {}".format(targetfpath))
    plt.show()

    # TODO: average and overlap

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: <path to model> <path to spectrogram image directory>")
        exit(1)
    elif not os.path.isfile(sys.argv[1]):
        print("{} is not a valid file. Need a path to a trained VAE.".format(sys.argv[1]))
        exit(2)
    elif not os.path.isdir(sys.argv[2]):
        print("{} is not a valid directory. Need a path to a directory of preprocessed spectrograms.".format(sys.argv[2]))
        exit(3)

    # Load the configuration
    configfpath = os.path.abspath("../../Artie/experiment/configfiles/testthesis.cfg")
    config = configuration.load(None, fpath=configfpath)

    # Random seed
    np.random.seed(1263262)

    # Load the VAE
    autoencoder = p1._build_vae(config)
    autoencoder.load_weights(sys.argv[1])

    # Load a bunch of spectrograms into a batch
    specs = load_spectrograms_from_directory(sys.argv[2])
    nspecs = specs.shape[0]
    #specs = np.expand_dims(specs[0, :, :, :], 0)

    # Run the batch to get the encodings
    batchsize = config.getint('autoencoder', 'batchsize')

    try:
        # The output of the encoder portion of the model is three items: Mean, LogVariance, and Value sampled from described distribution
        means, logvars, encodings = autoencoder._encoder.predict(specs, batch_size=None, steps=1)
    except Exception:
        print("Probably out of memory. Trying as an image generator instead.")
        specs = None  # Hint to the GC

        # Remove the useless subdirectory from the path (the imagedatagen needs it, but can't be told about it... ugh)
        pathsplit = sys.argv[2].rstrip(os.sep).split(os.sep)
        root = os.path.join(*[os.sep if p == '' else p for p in pathsplit[0:-1]])
        nworkers = config.getint('autoencoder', 'nworkers')
        imshapes = config.getlist('autoencoder', 'input_shape')[0:2]  # take only the first two dimensions (not channels)
        imshapes = [int(i) for i in imshapes]
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
        means, logvars, encodings = autoencoder._encoder.predict_generator(datagen,
                                            steps=int(nspecs / batchsize),
                                            use_multiprocessing=False,
                                            workers=nworkers)

    stdevs = np.exp(0.5 * logvars)
    # Visualize where the encodings ended up

    # Plot where each encoding is
    plt.scatter(encodings[:, 0], encodings[:, 1])
    plt.title("Scatter Plot of Encodings")
    plt.show()

    # Plot the distributions as circles whose means determine location and whose radii are composed
    # of the standard deviations
    plt.scatter(means[:, 0], means[:, 1], s=np.square(stdevs * 10))
    plt.title("Distributions the Encodings were drawn From")
    plt.show()
