"""
This is the main entry point for the ArtieInfant experiment.
"""
import os
import sys
import instinct
import internals
import output
import senses

if __name__ == "__main__":
    # Phase 1
    # TODO: pretrain RL algo to vocalize
    # TODO: train VAE
    # TODO: Mean Shift Cluster on the VAE latent vectors to determine number of clusters ('phonemes')
    # TODO: Determine prototype WAV for each phoneme
    # TODO: reinforce RL agent based on utterances similar to prototypes

    # Phase 2
    # TODO: train language model (RNN)
    #       -> Sometimes output through the decoder

    # Phase 3
    # TODO: tie everything together

    # Phase 4
    # TODO: babble
    pass
