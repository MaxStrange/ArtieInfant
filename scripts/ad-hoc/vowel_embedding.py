"""
Loads an autoencoder model, plots the latent space for the test set and the
plots sounds from a directory on top of that space, with arrows pointing to
each of the overlaid sounds. The arrows have labels that are the file names
of the sounds (without the extension).
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Load the stuff we need from ArtieInfant proper
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../Artie"))
from experiment.thesis import phase1                    # pylint: disable=locally-disabled, import-error
from experiment.analysis.vae import plotvae             # pylint: disable=locally-disabled, import-error


def _plot(test_embeddings: np.ndarray, special_embeddings: np.ndarray, special_labels: [str], ndims: int) -> None:
    """
    Plots the given embeddings and labels.
    """
    fig = plt.figure()

    if ndims == 1:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X')
        ax.scatter(test_embeddings, np.zeros_like(test_embeddings))
        ax.scatter(special_embeddings, np.zeros_like(special_embeddings), c='red')
    elif ndims == 2:
        ax = fig.add_subplot(111)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.scatter(test_embeddings[:, 0], test_embeddings[:, 1])
        ax.scatter(special_embeddings[:, 0], special_embeddings[:, 1], c='red')
    elif ndims == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.scatter(test_embeddings[:, 0], test_embeddings[:, 1], test_embeddings[:, 2])
        ax.scatter(special_embeddings[:, 0], special_embeddings[:, 1], special_embeddings[:, 2], c='red')
    else:
        raise ValueError("`ndims` must be 1, 2, or 3, but is {}".format(ndims))

    ax.set_title("Scatter Plot of Embeddings")

    save = "scatter_embeddings_ad_hoc.png"
    print("Saving", save)
    plt.savefig(save)
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('aemodelpath', type=str, help="Path to the Auto Encoder weights.")
    parser.add_argument('specmode', choices=['long', 'short'], help="Long: 241x20x1 spectrograms; Short: 81x18x1")
    parser.add_argument('overlaydir', type=str, help="Directory that contains the sound files you want to overlay on the test set's embeddings")
    parser.add_argument('--ndims', default=3, type=int, help="The number of dimensions of the latent space for the given model of autoencoder.")
    args = parser.parse_args()

    # Validate args
    if not os.path.isfile(args.aemodelpath):
        print("Not a file: {}".format(args.aemodelpath))
        exit(1)
    elif not os.path.isdir(args.overlaydir):
        print("Not a directory: {}".format(args.overlaydir))
        exit(2)

    # Set stuff up based on what mode we are
    if args.specmode == 'long':
        input_shape = [241, 20, 1]
        testdir = "/home/max/Dropbox/thesis/harddrive_backup/test_spectrogram_images/test_set"
        sample_rate_hz = 16000.0
        duration_s = 0.5
        window_length_s = 0.03
        ae = phase1._build_vae1(is_variational=False, input_shape=input_shape, latent_dim=args.ndims, optimizer='adadelta', loss='mse', tbdir=None, kl_loss_prop=None, recon_loss_prop=None, std_loss_prop=None)
    else:
        input_shape = [81, 18, 1]
        testdir = "/home/max/Dropbox/thesis/harddrive_backup/filterbank_images/test_set"
        sample_rate_hz = 8000.0
        duration_s = 0.3
        window_length_s = 0.02
        ae = phase1._build_vae2(is_variational=False, input_shape=input_shape, latent_dim=args.ndims, optimizer='adadelta', loss='mse', tbdir=None, kl_loss_prop=None, recon_loss_prop=None, std_loss_prop=None)

    # Load the weights into the autoencoder
    ae.load_weights(args.aemodelpath)

    # Encode the test set
    _, _, test_set_embeddings = plotvae._predict_on_spectrograms(testdir, ae, batchsize=32, nworkers=4, imshapes=input_shape)

    # Encode the audio files found in the directory
    _, _, special_embeddings, labels = plotvae._predict_on_sound_files(fpaths=None, dpath=args.overlaydir, model=ae, sample_rate_hz=sample_rate_hz, duration_s=duration_s, window_length_s=window_length_s)

    # Now plot the embedding space
    _plot(test_set_embeddings, special_embeddings, labels, args.ndims)
