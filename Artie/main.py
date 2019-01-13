"""
This is the main entry point for the ArtieInfant experiment.
"""
from experiment.thesis import phase1
import argparse
import os
import sys
import instinct
import internals
import logging
import output
import senses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Runs the experiment using the test configuration.")
    parser.add_argument("--loglevel", choices=["warn, info, debug"], default="debug", help="Log level for debug logging during the experiment.")
    parser.add_argument("--logfile", type=str, default="experimentlog.log", help="Path to the log file to write logs to.")
    parser.add_argument("--preprocess", action="store_true", help="Preprocesses all the data as part of training.")
    args = parser.parse_args()

    # Set up the logging configuration
    loglevel = getattr(logging, args.loglevel.upper())
    logging.basicConfig(filename=args.logfile, filemode='w', level=loglevel)

    # Phase 1
    phase1.run(preprocess=args.preprocess, test=args.test)

    # Phase 2
    # TODO: train language model (RNN)
    #       -> Sometimes output through the decoder

    # Phase 3
    # TODO: tie everything together

    # Phase 4
    # TODO: babble
