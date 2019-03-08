"""
This script is (was) for tuning the genetic algorithm for mimicking sounds.
"""
import audiosegment as asg
import argparse
import os
import sys

sys.path.append(os.path.abspath("../../Artie/experiment"))
sys.path.append(os.path.abspath("../../Artie"))
import experiment.configuration as configuration                # pylint: disable=locally-disabled, import-error
import internals.motorcortex.motorcortex as mc                  # pylint: disable=locally-disabled, import-error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="The target WAV file to mimic")
    parser.add_argument("-p", "--pretrain", action="store_true", help="Should we pretrain to make noise before training to mimic?")
    args = parser.parse_args()

    # Load the configuration file
    config = configuration.load("Tuning", fpath="tuneconfig.cfg")

    # Load the target wav file
    target = asg.from_file(args.target)

    # Build the model and train
    model = mc.SynthModel(config)
    if args.pretrain:
        print("Pretraining...")
        model.pretrain()

    model.train(target, savefpath="Phase1Output.wav")
