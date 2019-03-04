"""
Load the given spectrogram model, run a bunch of spectrograms through it,
then see what it does with them in its 2D latent space.
"""
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("../../Artie/experiment"))
sys.path.append(os.path.abspath("../../Artie"))
import thesis.phase1 as p1                                      # pylint: disable=locally-disabled, import-error
import experiment.configuration as configuration                # pylint: disable=locally-disabled, import-error

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
    return np.expand_dims(np.array(specs), -1)

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

    # Load the VAE
    autoencoder = p1._build_vae(config)

    # Load a bunch of spectrograms into a batch
    specs = load_spectrograms_from_directory(sys.argv[2])
    #specs = np.expand_dims(specs[0, :, :, :], 0)

    # Run the batch to get the encodings
    batchsize = config.getint('autoencoder', 'batchsize')
    # The output of the encoder portion of the model is three items: Mean, LogVariance, and Value sampled from described distribution
    means, logvars, encodings = autoencoder._encoder.predict(specs, batch_size=None, steps=1)
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
