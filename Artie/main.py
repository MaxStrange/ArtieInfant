"""
This is the main entry point for the ArtieInfant experiment.
"""
from experiment import configuration
from experiment.thesis import phase1
import instinct
import internals
import senses

import argparse
import logging
import numpy as np
import os
import output
import shutil
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="The path to the config file to use.")
    parser.add_argument("--loglevel", choices=["warn", "info", "debug"], default="debug", help="Log level for debug logging during the experiment.")
    parser.add_argument("--logfile", type=str, default="experimentlog.log", help="Path to the log file to write logs to.")
    parser.add_argument("--preprocess", action="store_true", help="Preprocesses all the data as part of training.")
    parser.add_argument("--pretrain-synth", action="store_true", help="Pretrain the voice synthesizer.")
    parser.add_argument("--spectrograms", action="store_true", help="Convert preprocessed data into spectrogram images.")
    parser.add_argument("--train-vae", action="store_true", help="Train the Variational Autoencoder. If not, we will attempt to load a pretrained one.")
    parser.add_argument("--train-synth", action="store_true", help="Train the voice synthesizer. If not, we will attempt to load a pretrained one.")
    args = parser.parse_args()

    # Set up the logging configuration
    loglevel = getattr(logging, args.loglevel.upper())
    logging.basicConfig(filename=args.logfile, filemode='w', level=loglevel)

    # Load the correct config file
    config = configuration.load(args.config)

    # Make a folder for the analysis results
    experimentname = config.getstr('experiment', 'name')
    saveroot = config.getstr('experiment', 'save_root')
    savedir = os.path.join(saveroot, experimentname)
    os.makedirs(savedir, exist_ok=True)

    # Random seed
    randomseed = config.getint('experiment', 'random-seed')
    np.random.seed(randomseed)

    # Phase 1
    phase1.run(config, savedir,
                preprocess=args.preprocess,
                preprocess_part_two=args.spectrograms,
                pretrain_synth=args.pretrain_synth,
                train_vae=args.train_vae,
                train_synth=args.train_synth)

    # Phase 2
    # TODO: train language model (RNN)
    #       -> Sometimes output through the decoder

    # Phase 3
    # TODO: tie everything together

    # Phase 4
    # TODO: babble

    # Put a copy of the log file into the save directory
    shutil.copyfile(args.logfile, os.path.join(savedir, os.path.basename(args.logfile)))
