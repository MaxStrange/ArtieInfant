"""
Load the given spectrogram model and test it on an input image, showing the image and autoencoded image.

Also shows a sampling from latent space.
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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: <path to model> <path to spectrogram image>")
        exit(1)
    elif not os.path.isfile(sys.argv[1]):
        print("{} is not a valid file. Need a path to a trained VAE.".format(sys.argv[1]))
        exit(2)
    elif not os.path.isfile(sys.argv[2]):
        print("{} is not a valid file. Need a path to a preprocessed spectrogram.".format(sys.argv[2]))
        exit(3)

    # Create the configuration
    configfpath = os.path.abspath("../../Artie/experiment/configfiles/testthesis.cfg")
    config = configuration.load(None, fpath=configfpath)

    # Load the VAE
    autoencoder = p1._build_vae(config)

    # Load the spectrogram
    spec = imageio.imread(sys.argv[2])
    spec = spec / 255.0
    assert len(spec.shape) == 2, "Shape of spectrogram before encode/decode is {}. Expected (nrows, ncols)".format(spec)

    # Run the spectrogram through the VAE
    decoded_spec = autoencoder.encode_decode(spec)
    assert len(decoded_spec.shape) == 4, "Shape of decoded spectrogram is {}. Expected (batch, nrows, ncols, nchannels)".format(decoded_spec.shape)
    assert decoded_spec.shape[-1] == 1, "Expected nchannels to be the last item in the decoded spectrogram's shape, but it isn't."
    decoded_spec = np.reshape(decoded_spec, spec.shape)

    # Make up some bogus frequencies and times
    fs = [f for f in range(0, spec.shape[0])]
    ts = [t for t in range(0, spec.shape[1])]

    # Display before and after
    plt.title("Before (left) and After (right)")
    plt.subplot(121)
    plt.pcolormesh(ts, fs, spec)
    plt.subplot(122)
    plt.pcolormesh(ts, fs, decoded_spec)
    plt.show()

    # Take a few samples from latent space and see what we get
    nlatentdims = config.getint('autoencoder', 'nembedding_dims')
    nsamples = 4
    for subpltidx in range(1, nsamples + 1):
        z = [autoencoder.sample()]
        sample = np.reshape(autoencoder.predict([z]), spec.shape)
        plt.subplot(100 + (nsamples * 10) + subpltidx)
        plt.title(z)
        plt.pcolormesh(ts, fs, sample)
    plt.show()

    # If we have a 2D embedding space, let's vary each dimension and plot a grid
    if nlatentdims == 2:
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        n = 8
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        k = 1
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = autoencoder.predict(z_sample)
                sample = np.reshape(x_decoded, spec.shape)
                plt.subplot(n, n, k)
                plt.pcolormesh(ts, fs, sample)
                k += 1
        plt.show()
