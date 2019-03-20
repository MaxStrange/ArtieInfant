"""
Load the given spectrogram model and test it on an input image, showing the image and autoencoded image.

Also shows a sampling from latent space.
"""
import argparse
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath("../../Artie/experiment"))
sys.path.append(os.path.abspath("../../Artie"))
import thesis.phase1 as p1                                      # pylint: disable=locally-disabled, import-error
import experiment.configuration as configuration                # pylint: disable=locally-disabled, import-error

def plot_stats_of_embeddings_for_wav_file(audiofpath):
    """
    Prints and plots some interesting stuffs regarding the embeddings generated from
    spectrograms sliding across the given audio file.
    """
    assert os.path.isfile(audiofpath), "{} is not a valid file path.".format(audiofpath)

    # TODO:
    # Load the file into memory
    # Slice it according to config file
    # Compute spectrograms from each slice
    # Load the VAE
    # Compute embeddings from the encoder portion for each spectrogram
    raise NotImplementedError("Future Self! remember to do this!")

def _validate_args(args):
    """
    Exits after printing a helpful error message if any of the args don't make sense.
    """
    if not os.path.isfile(args.model):
        print("{} is not a valid file. Need a valid trained VAE weights file.".format(args.model))
        exit(1)
    if args.image:
        for ipath in args.image:
            if not os.path.isfile(ipath):
                print("{} is not a valid path to a spectrogram.".format(ipath))
                exit(2)
    if args.topo:
        low, high = args.topo
        if low >= high:
            print("Low ({}) must be less than high ({})".format(low, high))
            exit(3)

def _plot_input_output_spectrograms(ipath, autoencoder):
    """
    Loads `ipath` into a spectrogram and then runs it through
    `autoencoder`. Plots the input on the left and the output
    on the right.

    Returns the shape of the spectrograms so other functions
    don't have to figure it out.
    """
    spec = imageio.imread(ipath)
    spec = spec / 255.0  # ImageDataGenerator rescale factor of 1.0/255.0
    assert len(spec.shape) == 2, "Shape of spectrogram before encode/decode is {}. Expected (nrows, ncols)".format(spec)

    # Run the spectrogram through the VAE
    decoded_spec = autoencoder.encode_decode(spec)
    assert len(decoded_spec.shape) == 4, "Shape of decoded spectrogram is {}. Expected (batch, nrows, ncols, nchannels)".format(decoded_spec.shape)
    assert decoded_spec.shape[-1] == 1, "Expected nchannels to be the last item in the decoded spectrogram's shape, but it isn't."
    decoded_spec = np.reshape(decoded_spec, spec.shape) * 255.0

    # Make up some bogus frequencies and times
    fs = [f for f in range(0, spec.shape[0])]
    ts = [t for t in range(0, spec.shape[1])]

    # Display before and after
    plt.title("Before (left) and After (right) for {}".format(os.path.basename(ipath)))
    plt.subplot(121)
    plt.pcolormesh(ts, fs, spec)
    plt.subplot(122)
    plt.pcolormesh(ts, fs, decoded_spec)
    name = os.path.splitext(os.path.basename(ipath))[0]
    plt.savefig("spectrogram_{}.png".format(name))
    plt.show()

    return spec.shape

def _plot_samples_from_latent_space(autoencoder, shape):
    """
    Samples embeddings from latent space and decodes them. Then plots them.

    `shape` is the shape of the spectrogram that we will be getting out of
    the decoder.
    """
    # Make up some bogus frequencies and times
    fs = [f for f in range(0, shape[0])]
    ts = [t for t in range(0, shape[1])]

    # List of distros to sample from (mu, sigma)
    # Change this to suit your needs
    distros = [
        ((3.7, -3.7), (1.2, 0.4)),
        ((2.8, -2.8), (1.0, 0.3)),
        ((2.3, -2.0), (2.0, 1.5)),
        ((5.0, -1.0), (4.0, 0.5)),
    ]

    # Go through each distro and sample from it several times
    # Decode the samples
    nsamples = 6
    for j, (mu, sigma) in enumerate(distros):
        for i in range(1, nsamples):
            z = [autoencoder.sample_from_gaussian(mu, sigma)]
            sample = np.reshape(autoencoder.predict([z]), shape)
            plt.subplot(len(distros), nsamples, i + (j * nsamples))
            plt.title("({:.2f},{:.2f})".format(z[0][0], z[0][1]))
            plt.pcolormesh(ts, fs, sample * 255.0)
    # Plot everything
    plt.show()

def _plot_topographic_swathe(autoencoder, shape, low, high):
    """
    Plot a topographic swathe; square from low to high in x and y.
    Only works if we have a 2D embedding space (in 3D we would need a cube,
    and beyond that is impossible to visualize intuitively).
    """
    n = 8
    grid_x = np.linspace(low, high, n)
    grid_y = np.linspace(low, high, n)[::-1]

    # Make up some bogus frequencies and times
    fs = [f for f in range(0, shape[0])]
    ts = [t for t in range(0, shape[1])]

    k = 1
    for yi in grid_y:
        for xi in grid_x:
            z_sample = np.array([[xi, yi]])
            x_decoded = autoencoder.predict(z_sample) * 255.0
            sample = np.reshape(x_decoded, shape)
            plt.subplot(n, n, k)
            plt.pcolormesh(ts, fs, sample)
            k += 1
    plt.savefig("spectrogram_swathe_{:.1f}_{:.1f}.png".format(low, high))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to VAE model weights for the current architecture")
    parser.add_argument("-i", "--image", action="append", help="A spectogram image to try to recreate")
    parser.add_argument("-t", "--topo", nargs=2, type=float, help="(low, high). Show a topographical swathe between low and high.")
    args = parser.parse_args()

    _validate_args(args)

    # Load the configuration
    configfpath = os.path.abspath("../../Artie/experiment/configfiles/testthesis.cfg")
    config = configuration.load(None, fpath=configfpath)

    # Random seed
    np.random.seed(643662)

    # Load the VAE
    autoencoder = p1._build_vae(config)
    autoencoder.load_weights(sys.argv[1])

    # Load the spectrograms and run the model over them
    for ipath in args.image:
        shape = _plot_input_output_spectrograms(ipath, autoencoder)

    # Take a few samples from latent space just to see what we get
    _plot_samples_from_latent_space(autoencoder, shape)

    # If we have a 2D embedding space, let's vary each dimension and plot a grid
    nlatentdims = config.getint('autoencoder', 'nembedding_dims')
    if args.topo and nlatentdims == 2:
        _plot_topographic_swathe(autoencoder, shape, args.topo[0], args.topo[1])
