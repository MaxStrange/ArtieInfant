"""
This is the main entry point for the ArtieInfant experiment.
"""
import backend.kerasback as kerasbackend            # pylint: disable=locally-disabled, no-name-in-module
import backend.pytorchback as pytorchbackend        # pylint: disable=locally-disabled, no-name-in-module
import experiment.configuration as configuration
import experiment.thesis.phase1 as phase1
import instinct
import internals
import senses

import argparse
import logging
import numpy as np
import os
import output
import random
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
    parser.add_argument("--back-end", choices=("keras", "pytorch"), default='keras', help="Should we use Keras or PyTorch?")
    args = parser.parse_args()

    # Set up the logging configuration
    loglevel = getattr(logging, args.loglevel.upper())
    logging.basicConfig(filename=args.logfile, filemode='w', level=loglevel)

    # Get the appropriate backend
    if args.back_end == "keras":
        nnbackend = kerasbackend
    elif args.back_end == "pytorch":
        nnbackend = pytorchbackend
    else:
        raise NotImplementedError("Backend {} is not yet implemented.".format(args.back_end))

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
    random.seed(randomseed)
    nnbackend.seed(randomseed)
    # Note that due to using a GPU and using multiprocessing, reproducibility is not guaranteed
    # But the above lines do their best

    # Phase 1
    phase1.run(config, savedir,
                preprocess=args.preprocess,
                preprocess_part_two=args.spectrograms,
                pretrain_synth=args.pretrain_synth,
                train_vae=args.train_vae,
                train_synth=args.train_synth,
                network_backend=nnbackend)

    # Phase 2
    # TODO: train language model (RNN)
    #       -> Sometimes output through the decoder

    # Phase 3
    # TODO: tie everything together

    # Phase 4
    # TODO: babble

    # Put a copy of the log file into the save directory
    shutil.copyfile(args.logfile, os.path.join(savedir, os.path.basename(args.logfile)))
