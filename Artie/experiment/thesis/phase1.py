"""
This is the phase 1 file.

This file's API consists simply of the function run(), which will run phase 1 of the thesis experiment.
"""
from internals.specifics import rl    # pylint: disable=locally-disabled, import-error
from experiment import configuration  # pylint: disable=locally-disabled, import-error
import os

def run(test=False):
    """
    Entry point for Phase 1.

    Initializes and pretrains the voice synthesization network to vocalize;
    Creates and trains a variational autoencoder on the entire Oliver data set;
    Applies a clustering algorithm to the embeddings that the VAE comes up with once it is trained;
    Determines a prototype sound for each cluster;
    Finishes training the voice synthesizer to mimic these sounds based on which embedding it observes.

    If `test` is True, we will load the testthesis.cfg config file instead of the thesis config.
    """
    # Load the right experiment configuration
    configname = "testthesis" if test else "thesis"
    config = configuration.load(configname)

    # Pretrain the voice synthesizer to make non-specific noise
    weightpathbasename, actor, critic = rl.pretrain(config)

    # TODO:
    #   VAE - train then run over a suitable sample of audio to save enough embeddings for the prototypes/clustering
    #       # Train the VAE to a suitable level of accuracy
    #       # Use the trained VAE on ~1,000 (or more?) audio samples, saving each audio sample along with its embedding.
    #   Mean Shift Cluster - cluster the saved embeddings using sklearn.mean_shift_cluster (or whatever it's called). This will tell us how many clusters.
    #       # Load the saved embeddings into a dataset
    #       # Run the SKLearn algorithm over the dataset to determine how many clusters and to get cluster indexes.
    #   Determine prototypes - Go through and send each embedding into the clusterer to get its cluster index. Take one from each cluster to form a prototype.
    #       # Determine each saved embedding's cluster index.
    #       # Take 'quintessential' embeddings from each cluster and save them as prototypes.
    #   Finish training rl agent - Set up the environment with these prototypes and the weights of the pretrained agent. Train until it can mimic the prototypes given a cluster index.
    #       # Train the RL agent to mimic the given prototypes
    #   Inference: Save the trained RL agent and then you can use it to make noises based on cluster index inputs.
    #   You now have trained this:
    #   [1]  ->  /a/
    #   [2]  ->  /g/
    #   [.]  ->  ..
    #   [9]  ->  /b/
    #   [.]  ->  ..
    #   Which means that you now have a map of phonemes.
    #   You also have an embedding space that can be sampled from. That sample could then be run through the clusterer to determine
    #   the index of the sample, which would then determine which sound it was. I'm not sure what this gives you... but it seems like it might be important.

    # Clean up the weights
    os.remove(weightpathbasename + "_actor" + ".hdf5")
    os.remove(weightpathbasename + "_critic" + ".hdf5")
