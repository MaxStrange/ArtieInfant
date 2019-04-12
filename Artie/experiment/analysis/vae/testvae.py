"""
Load the given spectrogram model and test it on an input image, showing the image and autoencoded image.

Also shows a sampling from latent space.
"""
import argparse
import audiosegment as asg
import imageio
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys

from experiment.thesis import phase1 as p1                      # pylint: disable=locally-disabled, import-error
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

def _plot_input_output_spectrograms(audiofpath, ipath, autoencoder, savedir, window_length_s, overlap):
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

    # Make up some bogus frequencies and times  # TODO
    # Load in the audio file and do the spectrogram to get the right frequencies and times
    seg = asg.from_file(audiofpath)
    fs, ts, _amps = seg.spectrogram(window_length_s=window_length_s, overlap=overlap, window=('tukey', 0.5))

    # Calculate the MSE of the two
    reconloss = np.sum(np.square(spec - decoded_spec))

    # Display before and after
    msg = "Reconstructive loss for {}: {}".format(ipath, reconloss)
    print(msg)
    logging.info(msg)

    # Plot side-by-side
    fig, axs = plt.subplots(1, 2, squeeze=True)
    fig.suptitle("Before (left) and After (right) for {}".format(os.path.basename(ipath)))
    fig.text(0.5, 0.04, "Time (s)", ha='center', va='center')

    axs[0].pcolormesh(ts, fs, spec)
    axs[0].set_ylabel("Hz")
    #axs[0].set_xlabel("Time (s)")

    axs[1].pcolormesh(ts, fs, decoded_spec)
    axs[1].yaxis.set_ticklabels([])
    #axs[1].set_xlabel("Time (s)")

    # Save the figure
    name = os.path.splitext(os.path.basename(ipath))[0]
    save = os.path.join(savedir, "spectrogram_{}.png".format(name))
    print("Saving", save)
    plt.savefig(save)
    plt.clf()

    # Now also save the audio file that corresponds to this image
    save = os.path.join(savedir, os.path.basename(audiofpath))
    shutil.copyfile(audiofpath, save)

    return fs, ts

def _plot_samples_from_latent_space(autoencoder, shape, savedir, frequencies, times, ndims=2):
    """
    Samples embeddings from latent space and decodes them. Then plots them.

    `shape` is the shape of the spectrogram that we will be getting out of
    the decoder.
    """
    assert ndims in (1, 2, 3), "This visualization is impossible with ndims > 3. Is: {}".format(ndims)

    # List of distros to sample from (ndimensional mu, ndimensional sigma)
    distros = [(np.random.normal(0.0, 2.5, size=ndims), np.abs(np.random.normal(1.0, 0.5, size=ndims))) for _ in range(4)]

    # Go through each distro and sample from it several times
    # Decode the samples
    nsamples = 6
    fig, axs = plt.subplots(len(distros), nsamples, constrained_layout=True)
    for j, (mu, sigma) in enumerate(distros):
        for i in range(nsamples):
            # Sample from the autoencoder's latent space
            z = [autoencoder.sample_from_gaussian(mu, sigma)]
            sample = np.reshape(autoencoder.predict([z]), shape)

            # Plot the sample into the grid of figures
            axs[j][i].pcolormesh(times, frequencies, sample * 255.0)
            if ndims == 1:
                axs[j][i].set_title("{:.2f}".format(z[0][0]))
                axs[j][i].title.set_fontsize(8)
            elif ndims == 2:
                axs[j][i].set_title("({:.2f},{:.2f})".format(z[0][0], z[0][1]))
                axs[j][i].title.set_fontsize(8)
            elif ndims == 3:
                axs[j][i].set_title("({:.2f},{:.2f},{:.2f})".format(z[0][0], z[0][1], z[0][2]))
                axs[j][i].title.set_fontsize(8)

            if j == len(distros) - 1:
                # We are on the last row, let's label the x axes
                pass
            else:
                # If we aren't on the last row, let's not have xticks
                axs[j][i].xaxis.set_ticklabels([])

            if i == 0:
                # We are on the left, let's label the y axes
                pass
            else:
                # If we aren't on the left, let's not have yticks
                axs[j][i].yaxis.set_ticklabels([])

    fig.suptitle("Samples from Latent Space")
    #fig.text(0.5, 0.04, "Time (s)", ha='center', va='center')
    fig.text(0.03, 0.5, "Hz", ha='center', va='center', rotation='vertical')

    # Plot everything
    save = os.path.join(savedir, "samples_from_latent_space.png")
    print("Saving", save)
    plt.savefig(save)
    plt.clf()

def _plot_topographic_swathe(autoencoder, shape, low, high, savedir, frequencies, times, ndims=2):
    """
    Plot a topographic swathe; square from low to high in x and y.
    Only works if we have a 1D or 2D embedding space (in 3D we would need a cube,
    and beyond that is impossible to visualize intuitively).
    """
    n = 8
    grid_x = np.linspace(low, high, n)
    grid_y = np.linspace(low, high, n)[::-1]

    if ndims == 1:
        fig, axs = plt.subplots(1, n)
        for k, x in enumerate(grid_x):
            z_sample = np.array([x])
            x_decoded = autoencoder.predict(z_sample) * 255.0
            sample = np.reshape(x_decoded, shape)
            axs[k].pcolormesh(times, frequencies, sample)
            axs[k].set_xlabel("Time (s)")
            if k == 0:
                axs[k].set_ylabel("Hz")
            else:
                axs[k].yaxis.set_ticklabels([])
    elif ndims == 2:
        fig, axs = plt.subplots(len(grid_y), len(grid_x))
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = autoencoder.predict(z_sample) * 255.0
                sample = np.reshape(x_decoded, shape)
                axs[i][j].pcolormesh(times, frequencies, sample)
                if j == 0:
                    # We are on the left
                    #axs[i][j].set_ylabel("Hz")
                    pass
                else:
                    axs[i][j].yaxis.set_ticklabels([])

                if i == len(grid_y) - 1:
                    # We are on the bottom row
                    #axs[i][j].set_xlabel("Time (s)")
                    pass
                else:
                    axs[i][j].xaxis.set_ticklabels([])
    else:
        raise ValueError("Cannot plot a topographic swathe for dimensions higher than 2 currently. Passed ndims={}".format(ndims))
    fig.suptitle("Topographic Swathe")
    fig.text(0.5, 0.04, "Time (s)", ha='center', va='center')
    fig.text(0.03, 0.5, "Hz", ha='center', va='center', rotation='vertical')

    save = os.path.join(savedir, "spectrogram_swathe_{:.1f}_{:.1f}.png".format(low, high))
    print("Saving", save)
    plt.savefig(save)
    plt.clf()
