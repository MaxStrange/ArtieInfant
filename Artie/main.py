"""
This is the main entry point for the ArtieInfant experiment.
"""
from experiment.thesis import phase1
import argparse
import os
import sys
import instinct
import internals
import output
import senses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Runs the experiment using the test configuration.")
    args = parser.parse_args()

    # Phase 1
    phase1.run(test=args.test)

    # Phase 2
    # TODO: train language model (RNN)
    #       -> Sometimes output through the decoder

    # Phase 3
    # TODO: tie everything together

    # Phase 4
    # TODO: babble
