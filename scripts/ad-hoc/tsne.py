"""
This script visualizes the high-dimensional embedding spaces using T-SNE,
which maps high dimensional data into lower dimensions while maintaining
a given distance metric (which, for our purposes, is pretty much all we care about).
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn
import sys

# Load the stuff we need from ArtieInfant proper
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../Artie"))
from experiment.thesis import phase1                    # pylint: disable=locally-disabled, import-error
from experiment.analysis.vae import plotvae             # pylint: disable=locally-disabled, import-error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('aemodelpath', type=str, help="Path to the Auto Encoder weights.")
    parser.add_argument('specmode', choices=['long', 'short'], help="Long: 241x20x1 spectrograms; Short: 81x18x1")
    parser.add_argument('ndims', type=int, help="The number of dimensions of the latent space for the given model of autoencoder.")
    parser.add_argument('--randseed', type=int, default=12543, help="Random seed for reproducing t-SNE results.")
    parser.add_argument('--perplexity', type=float, default=30.0, help="Perplexity of t-SNE. Values between 5 and 50 make sense typically.")
    parser.add_argument('--datadir', type=str, help="If given, we will convert all audio files in this directory into spectrograms and embed them in the resulting t-SNE plot.")
    args = parser.parse_args()

    np.random.seed(args.randseed)

    # Validate args
    if not os.path.isfile(args.aemodelpath):
        print("Not a file: {}".format(args.aemodelpath))
        exit(1)
    elif args.datadir and not os.path.isdir(args.datadir):
        print("Not a directory: {}".format(args.datadir))
        exit(2)

    # Parameters
    tsne_dimensions = 2

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
    ntest_embeddings = test_set_embeddings.shape[0]
    print(test_set_embeddings.shape)

    # Grab the special set if present
    if args.datadir:
        _, _, special_embeddings, special_names = plotvae._predict_on_sound_files(fpaths=None, dpath=args.datadir, model=ae, sample_rate_hz=sample_rate_hz, duration_s=duration_s, window_length_s=window_length_s)
        embeddings = np.append(test_set_embeddings, special_embeddings, axis=1)
    else:
        embeddings = test_set_embeddings
    print(embeddings.shape)

    # Do the T-SNE now and get back the embeddings that we will plot
    tsne = sklearn.manifold.TSNE(n_components=tsne_dimensions,
                                 perplexity=args.perplexity,    # Low values attempt to retain distance relationships over small distances (local), high means longer distances (global). The most reasonable values are 5 to 50
                                 learning_rate=200.0,           # try values between 10 and 1000 to adjust shape of outcome
                                 metric='euclidean',
                                 init='random',                 # try 'pca' if you want more stability
                                 verbose=2,
                                 angle=0.5)                     # try between 0.2 and 0.8. Not sure what it will do
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Do the plot
    if args.datadir:
        plotvae._plot_vanilla_latent_space(tsne_embeddings[0:ntest_embeddings, :], special_encodings=tsne_embeddings[ntest_embeddings:, :], name=None, savedir=".", ndims=tsne_dimensions, show=True)
    else:
        plotvae._plot_vanilla_latent_space(tsne_embeddings, special_encodings=None, name=None, savedir=".", ndims=tsne_dimensions, show=True)
