"""
This script is (was) for tuning the genetic algorithm for mimicking sounds.
"""
import audiosegment as asg
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sys

sys.path.append(os.path.abspath("../../Artie/experiment"))
sys.path.append(os.path.abspath("../../Artie"))
import experiment.configuration as configuration                # pylint: disable=locally-disabled, import-error
import internals.motorcortex.motorcortex as mc                  # pylint: disable=locally-disabled, import-error
import output.voice.synthesizer as synth                        # pylint: disable=locally-disabled, import-error

#################################################################
####### Globals ########
#################################################################
sample_rate_hz  = 16000.0    # 16kHz sample rate
bytewidth       = 2          # 16-bit samples
nchannels       = 1          # mono

nart_groups     = 5          # The number of articulator groups
#################################################################


def _plot_history(fname):
    """
    Reads in the CSV at `fname` and plots it as a history of the gene pools.
    """
    df = pandas.read_csv(fname)
    df = df.drop(['GenerationIndex'], axis=1)
    df.plot()
    plt.title(fname)
    plt.show()

def _load_audiofile(fname):
    """
    Load the given audio file and resample it to the correct parameters.
    """
    return asg.from_file(fname).resample(sample_rate_hz, bytewidth, nchannels).to_numpy_array().astype(float)

def _plot_wave_forms(waveforms):
    """
    Plots the given list of wave forms.
    """
    for i, wf in enumerate(waveforms):
        plt.subplot(len(waveforms), 1, i + 1)
        plt.plot(wf)
    plt.show()

def _plot_specs(pretrainfpath, trainfpath, targetset):
    """
    Plot the three audio segments as spectrograms side by side.
    """
    pretrain = asg.from_file(pretrainfpath).resample(sample_rate_hz, bytewidth, nchannels)
    train = asg.from_file(trainfpath).resample(sample_rate_hz, bytewidth, nchannels)

    # Plot the pretraining spectrogram
    fs, ts, amps = pretrain.spectrogram(0, 0.5, window_length_s=0.03, overlap=0.2, window=('tukey', 0.25))
    plt.subplot(1, 3, 1)
    plt.title("Pretraining")
    plt.pcolormesh(ts, fs, amps)

    # Plot the training spectrogram
    fs, ts, amps = train.spectrogram(0, 0.5, window_length_s=0.03, overlap=0.2, window=('tukey', 0.25))
    plt.subplot(1, 3, 2)
    plt.title("Training")
    plt.pcolormesh(ts, fs, amps)

    # Plot the target spectrogram
    fs, ts, amps = target.spectrogram(0, 0.5, window_length_s=0.03, overlap=0.2, window=('tukey', 0.25))
    plt.subplot(1, 3, 3)
    plt.title("Target")
    plt.pcolormesh(ts, fs, amps)

    plt.show()

def _plot_articulators(model):
    """
    Plot how each articulator changed over time in the genetic algorithm.
    """
    # Get the list of best agents
    bests = model.best_agents_phase1

    # Reshape each agent into a matrix of (articulator, time point)
    bests = [np.reshape(agent, (model._narticulators, -1)) for agent in bests]

    # Concatenate each matrix to (articulator, (time-points) * nagents)
    articulatormatrix = np.hstack(bests)
    print("Articulators over time:\n", articulatormatrix)

    # Plot the matrix, with the articulator names being the labels for the y axis
    ntimepoints = len(model._articulation_time_points_ms)
    ncols = int(articulatormatrix.shape[1] / ntimepoints)
    plt.title("Articulator Activations for Best Agents across Simulation")
    for row, art in enumerate(synth.articularizers):
        for col in range(ncols):
            plt.subplot(model._narticulators, ncols, (row * ncols) + col + 1)
            if col == 0:
                plt.ylabel(art, rotation='horizontal')
            plt.ylim(-1.0, 1.0)
            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
            start = col * ntimepoints
            end = start + ntimepoints
            plt.plot(articulatormatrix[row, start:end])
    plt.tight_layout()
    plt.show()

def analyze(model, target, ngroups):
    """
    Analyze the results of training the model. `target` is the resampled
    target AudioSegment that the mdoel has tried to learn to pronounce.

    `ngroups` is the number of articulator groups.
    """
    # Plot line graphs showing the progress through training
    _plot_history("Phase0OutputSound.csv")
    for i in range(ngroups):
        _plot_history("Phase1Output_{}.csv".format(i))

    # Load the output wave forms
    pretraining_output = _load_audiofile("Phase0OutputSound.wav")
    waveforms = [pretraining_output]
    for i in range(ngroups):
        training_output = _load_audiofile("Phase1Output_{}.wav".format(i))
        waveforms.append(training_output)

    target = target.to_numpy_array().astype(float)
    waveforms.append(target)

    # Show the raw outputs vs target
    _plot_wave_forms(waveforms)

    # Show the spectrogram representations of the three sounds
    _plot_specs("Phase0OutputSound.wav", "Phase1Output_{}.wav".format(ngroups - 1), target)

    # Plot how each articulator's activations changed over the course of the genetic algorithm
    #_plot_articulators(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="The target WAV file to mimic")
    parser.add_argument("-p", "--pretrain", action="store_true", help="Should we pretrain to make noise before training to mimic?")
    args = parser.parse_args()

    ## Load the configuration file
    config = configuration.load("Tuning", fpath="tuneconfig.cfg")

    ## Load the target wav file and resample
    target = asg.from_file(args.target).resample(sample_rate_hz, bytewidth, nchannels)

    # Build the model and train
    model = mc.SynthModel(config)
    if args.pretrain:
        print("Pretraining...")
        model.pretrain()

    print("Training...")
    model.train(target, savefpath="Phase1Output.wav")

    # Analyze stuff
    analyze(model, target, nart_groups)
