"""
Load the given spectrogram model and test it on an input image, showing the image and autoencoded image.

Also shows a sampling from latent space.
"""
import ae
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("USAGE: <path to model> <path to spectrogram image> [path to audiofile]")
        exit(1)
    elif not os.path.isfile(sys.argv[1]):
        print("{} is not a valid file. Need a path to a trained VAE.".format(sys.argv[1]))
        exit(2)
    elif not os.path.isfile(sys.argv[2]):
        print("{} is not a valid file. Need a path to a preprocessed spectrogram.".format(sys.argv[2]))
        exit(3)
    elif len(sys.argv) == 4 and not os.path.isfile(sys.argv[3]):
        print("{} is not a valid file. Need a path to a raw audio file.".format(sys.argv[3]))

    # Random seed
    np.random.seed(643662)

    # Load the VAE
    input_shape = (241, 20, 1)
    latent_dim = 2
    optimizer = 'adadelta'
    loss = 'mse'
    autoencoder = ae.cnn_vae(input_shape, latent_dim, optimizer, loss)
    autoencoder.load_weights(sys.argv[1])

    # Load the spectrogram
    spec = imageio.imread(sys.argv[2])
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
    plt.title("Before (left) and After (right)")
    plt.subplot(121)
    plt.pcolormesh(ts, fs, spec)
    plt.subplot(122)
    plt.pcolormesh(ts, fs, decoded_spec)
    plt.show()

    # Take a few samples from latent space and see what we get
    nsamples = 4
    for subpltidx in range(1, nsamples + 1):
        z = [autoencoder.sample()]  # Take z from the normal distribution
        sample = np.reshape(autoencoder.predict([z]), spec.shape)
        plt.subplot(100 + (nsamples * 10) + subpltidx)
        plt.title("From Normal Dist")
        plt.pcolormesh(ts, fs, sample * 255.0)
    plt.show()

    # If we have a 2D embedding space, let's vary each dimension and plot a grid
    if latent_dim == 2:
        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        n = 8
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        k = 1
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = autoencoder.predict(z_sample) * 255.0
                sample = np.reshape(x_decoded, spec.shape)
                plt.subplot(n, n, k)
                plt.pcolormesh(ts, fs, sample)
                k += 1
        plt.show()

    if len(sys.argv) == 4:
        audiofpath = sys.argv[3]
        plot_stats_of_embeddings_for_wav_file(audiofpath)
