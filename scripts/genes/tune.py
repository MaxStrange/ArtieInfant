"""
This script is (was) for tuning the genetic algorithm for mimicking sounds.
"""
import audiosegment as asg
import argparse
import os
import matplotlib.pyplot as plt
import pandas
import sys

sys.path.append(os.path.abspath("../../Artie/experiment"))
sys.path.append(os.path.abspath("../../Artie"))
import experiment.configuration as configuration                # pylint: disable=locally-disabled, import-error
import internals.motorcortex.motorcortex as mc                  # pylint: disable=locally-disabled, import-error

def spec(seg1, seg2):
    fs, ts, amps = seg1.spectrogram(0, 0.5, window_length_s=0.03, overlap=0.2, window=('tukey', 0.25))
    plt.subplot(1, 2, 1)
    plt.pcolormesh(ts, fs, amps)

    fs, ts, amps = seg2.spectrogram(0, 0.5, window_length_s=0.03, overlap=0.2, window=('tukey', 0.25))
    plt.subplot(1, 2, 2)
    plt.pcolormesh(ts, fs, amps)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="The target WAV file to mimic")
    parser.add_argument("-p", "--pretrain", action="store_true", help="Should we pretrain to make noise before training to mimic?")
    args = parser.parse_args()

    #################################################################
    sample_rate_hz  = 16000.0    # 16kHz sample rate
    bytewidth       = 2          # 16-bit samples
    nchannels       = 1          # mono
    #################################################################

    ## Load the configuration file
    config = configuration.load("Tuning", fpath="tuneconfig.cfg")

    ## Load the target wav file
    target = asg.from_file(args.target)
    target = target.resample(sample_rate_hz, bytewidth, nchannels)

    # Build the model and train
    model = mc.SynthModel(config)
    if args.pretrain:
        print("Pretraining...")
        model.pretrain()

    print("Training...")
    model.train(target, savefpath="Phase1Output.wav")
    df = pandas.read_csv("Phase1Output.csv")
    df = df.drop(['GenerationIndex'], axis=1)
    df.plot()
    plt.show()

    # Show the output from pretraining
    seg = asg.from_file("Phase0OutputSound.wav")
    seg = seg.resample(sample_rate_hz, bytewidth, nchannels)
    seg = seg.to_numpy_array().astype(float)
    plt.title("Phase 0 Output")
    plt.plot(seg)
    plt.show()

    seg = asg.from_file("Phase1Output.wav")
    seg = seg.resample(sample_rate_hz, bytewidth, nchannels)
    seg = seg.to_numpy_array().astype(float)

    # Show the raw output vs target
    plt.subplot(2, 1, 1)
    plt.title("Output Audio")
    plt.plot(seg)
    plt.subplot(2, 1, 2)
    plt.title("Target Audio")
    plt.plot(target.to_numpy_array().astype(float))
    plt.show()

    # Show the spectrogram representations of the two sounds
    spec(target, asg.from_file("Phase1Output.wav"))

    # TODO:
    # - Save the best agent from each generation
    # - Plot the best agent's values from each generation for each articulator
    # - Anneal:
    #   - Phase 0: Only the laryngeal articulators are allowed to move at all
    #       - After some number of steps, move their limits to be +- 0.1 of the best values found
    #   - Phase 1: Laryngeals now are only allowed to move that small amount. Each protophoneme has a population that goes through these steps:
    #       - Choose another articulator group and have it articulate for some number of steps
    #       - Take the best ones and anneal the limits to +- 0.1 of them
    #       - Repeat for another group of articulators
    #       - Repeat until all articulators have annealed for this population
