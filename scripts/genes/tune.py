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

    # Load the configuration file
    config = configuration.load("Tuning", fpath="tuneconfig.cfg")

    # Load the target wav file
    target = asg.from_file(args.target)
    target = target.resample(sample_rate_hz, bytewidth, nchannels)

    # Build the model and train
    model = mc.SynthModel(config)
    if args.pretrain:
        print("Pretraining...")
        model.pretrain()

    model.train(target, savefpath="Phase1Output.wav")
    df = pandas.read_csv("Phase1Output.csv")
    df = df.drop(['GenerationIndex'], axis=1)
    df.plot()
    plt.show()

    seg = asg.from_file("Phase1Output.wav")
    seg = seg.rsample(sample_rate_hz, bytewidth, nchannels)
    seg = seg.to_numpy_array().astype(float)

    plt.subplot(2, 1, 1)
    plt.title("Output Audio")
    plt.plot(seg)
    plt.subplot(2, 1, 2)
    plt.title("Input Audio")
    plt.plot(target.to_numpy_array().astype(float))
    plt.show()

    spec(target, asg.from_file("Phase1Output.wav"))
