"""
This script is (was) for tuning the genetic algorithm for mimicking sounds.
"""
import os
import sys

sys.path.append(os.path.abspath("../../Artie/experiment"))
sys.path.append(os.path.abspath("../../Artie"))
import experiment.configuration as configuration                # pylint: disable=locally-disabled, import-error
import internals.motorcortex.motorcortex as mc                  # pylint: disable=locally-disabled, import-error

if __name__ == "__main__":
    config = configuration.load("Tuning", fpath="tuneconfig.cfg")
    model = mc.SynthModel(config)
    #model.train()
