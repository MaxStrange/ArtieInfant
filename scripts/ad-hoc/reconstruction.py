"""
This script plots the reconstruction of all the spectrograms of the audio files in a directory,
given a particular model.
"""
import argparse
import audiosegment as asg
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import sys

# Load the stuff we need from ArtieInfant proper
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../Artie"))
from experiment.thesis import phase1                    # pylint: disable=locally-disabled, import-error
from experiment.analysis.vae import plotvae             # pylint: disable=locally-disabled, import-error

def _plot(specs, title, fname, times, frequencies):
    """
    Plot an nxn grid of spectrograms.
    """
    n = math.ceil(math.sqrt(specs.shape[0]))
    if n * (n - 1) >= specs.shape[0]:
        m = n - 1
    else:
        m = n
    #fig, axs = plt.subplots(n, n)
    fig = plt.figure()
    for i in range(m):
        for j in range(n):
            # Plot a subplot if possible
            if (i * n + j) < specs.shape[0]:
                sample = specs[i * n + j, :, :]
                ax = fig.add_subplot(m, n, (i * n + j) + 1)
                ax.pcolormesh(times, frequencies, sample)
            else:
                pass

            # Remove the ticklabels from y axis if we are not on the left
            if j == 0:
                # We are on the left
                pass
            else:
                ax.yaxis.set_ticklabels([])

            # Remove the ticklabels from x axis if we are not on the bottom
            if i == m - 1:
                # We are on the bottom row
                pass
            else:
                ax.xaxis.set_ticklabels([])

    fig.suptitle(title)
    fig.text(0.5, 0.04, "Time (s)", ha='center', va='center')
    fig.text(0.03, 0.5, "Hz", ha='center', va='center', rotation='vertical')

    save = os.path.join(os.path.abspath("."), fname)
    print("Saving", save)
    plt.savefig(save)
    plt.show()
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ndims', type=int, help="The number of dimensions of the latent space for the given model of autoencoder.")
    parser.add_argument('specmode', choices=['long', 'short'], help="Long: 241x20x1 spectrograms; Short: 81x18x1")
    parser.add_argument('aemodelpath', type=str, help="Path to the Auto Encoder weights.")
    parser.add_argument('audiodir', type=str, help="Directory that contains the sound files you want to reconstruct")
    args = parser.parse_args()

    # Validate args
    if not os.path.isfile(args.aemodelpath):
        print("Not a file: {}".format(args.aemodelpath))
        exit(1)
    elif not os.path.isdir(args.audiodir):
        print("Not a directory: {}".format(args.audiodir))
        exit(2)

    # Set stuff up based on what mode we are
    if args.specmode == 'long':
        input_shape = [241, 20, 1]
        sample_rate_hz = 16000.0
        duration_s = 0.5
        window_length_s = 0.03
        ae = phase1._build_vae1(is_variational=False, input_shape=input_shape, latent_dim=args.ndims, optimizer='adadelta', loss='mse', tbdir=None, kl_loss_prop=None, recon_loss_prop=None, std_loss_prop=None)
    else:
        input_shape = [81, 18, 1]
        sample_rate_hz = 8000.0
        duration_s = 0.3
        window_length_s = 0.02
        ae = phase1._build_vae2(is_variational=False, input_shape=input_shape, latent_dim=args.ndims, optimizer='adadelta', loss='mse', tbdir=None, kl_loss_prop=None, recon_loss_prop=None, std_loss_prop=None)

    # Load the weights into the autoencoder
    ae.load_weights(args.aemodelpath)

    # Load each file in the directory if it is a wav file
    wavpaths = [os.path.join(args.audiodir, fname) for fname in os.listdir(args.audiodir) if os.path.splitext(fname)[-1].lower() == ".wav"]
    segments = [asg.from_file(w) for w in wavpaths]
    segments = [seg.resample(sample_rate_Hz=sample_rate_hz) for seg in segments]

    # Turn each segment into a spectrogram (make sure to do exactly the same preprocessing as the autoencoder expects)
    spectrograms = [seg.spectrogram(0, duration_s, window_length_s=window_length_s, overlap=0.2) for seg in segments]
    amplitudes = [spec[-1] for spec in spectrograms]
    amplitudes = [amp * 255.0 / np.max(np.abs(amp)) for amp in amplitudes]
    amplitudes = [amp / 255.0 for amp in amplitudes]
    amplitudes = [np.expand_dims(amp, -1) for amp in amplitudes]
    amplitudes = np.array(amplitudes)

    # Run each one through the autoencoder to get the result
    reconstructions = ae.encode_decode(amplitudes)
    reconstructions = np.squeeze(reconstructions)
    amplitudes = np.squeeze(amplitudes)

    times = np.linspace(0, duration_s, num=input_shape[1])
    frequencies = np.linspace(0, sample_rate_hz / 2.0, num=input_shape[0])

    # Plot everything
    dirname = os.path.basename(args.audiodir)
    _plot(amplitudes, "Spectrograms of Variations on /{}/".format(dirname), "spectrogram_variations.png", times, frequencies)
    _plot(reconstructions, "Reconstructions of Variations on /{}/".format(dirname), "spectrogram_reconstruction.png", times, frequencies)
