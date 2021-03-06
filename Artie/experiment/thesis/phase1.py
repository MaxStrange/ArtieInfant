"""
This is the phase 1 file.

This file's API consists simply of the function run(), which will run phase 1 of the thesis experiment.
"""
from experiment.analysis import ae                      # pylint: disable=locally-disabled, import-error
from experiment.analysis import production              # pylint: disable=locally-disabled, import-error
from experiment.configuration import ConfigError        # pylint: disable=locally-disabled, import-error
from internals.motorcortex import motorcortex           # pylint: disable=locally-disabled, import-error
from internals.vae import vae                           # pylint: disable=locally-disabled, import-error
from internals.vae import ae as vanilla                 # pylint: disable=locally-disabled, import-error
from senses.voice_detector import voice_detector as vd  # pylint: disable=locally-disabled, import-error
from senses.dataproviders import sequence as seq        # pylint: disable=locally-disabled, import-error

from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import preprocessing

import audiosegment
import datetime
import imageio
import keras
import logging
import multiprocessing as mp
import numpy as np
import os
import random
import shutil
import sklearn
import tensorflow as tf
import tqdm
if "TRAVIS_CI" not in os.environ:
    import matplotlib.pyplot as plt

class VaeVisualizer(keras.callbacks.Callback):
    """
    Callback to pass into the VAE model when building it. This callback generates a before and after spectrogram
    sampled from a random batch after each epoch.
    """
    def __init__(self):
        # TODO: This is currently caching ALL of the data that we train on. We should not do that.
        super(VaeVisualizer, self).__init__()
        self.targets = []
        self.outputs = []
        self.inputs = []

        self.var_y_true = tf.Variable(0.0, validate_shape=False)
        self.var_y_pred = tf.Variable(0.0, validate_shape=False)
        self.var_x = tf.Variable(0.0, validate_shape=False)

    def on_batch_end(self, batch, logs=None):
        self.targets.append(K.eval(self.var_y_true))
        self.outputs.append(K.eval(self.var_y_pred))
        self.inputs.append(K.eval(self.var_x))

    def on_epoch_end(self, epoch, logs=None):
        # Get a random number to determine which batch to get
        batchidx = random.randint(0, len(self.inputs) - 1)

        # Use that number as an index into the batches from this epoch
        input_spec_batch = self.inputs[batchidx]

        # Get another random number
        idx = random.randint(0, input_spec_batch.shape[0] - 1)

        # Use that number as an index into the batch to get a random spectrogram
        input_spec = input_spec_batch[idx]

        # Get the times and frequencies (not the real ones, just some dummy values that we can feed into matplotlib)
        times = np.arange(0, input_spec.shape[1])
        freqs = np.arange(0, input_spec.shape[0])

        # Reshape the input spectrogram into the right shape for matplotlib
        inp = np.reshape(input_spec, (len(freqs), len(times)))

        # Plot the input spectrogram on the left (also modify the amplitudes to make them more visible)
        plt.subplot(121)
        plt.title("Input (batch, idx): {}".format((batchidx, idx)))
        plt.pcolormesh(times, freqs, inp)

        # Get the corresponding output spectrogram
        output_spec = self.outputs[batchidx][idx]

        # Reshape the output spectrogram into the right shape for matplotlib
        outp = np.reshape(output_spec, (len(freqs), len(times)))

        # Plot it on the right
        plt.subplot(122)
        plt.title("Output (batch, idx): {}".format((batchidx, idx)))
        plt.pcolormesh(times, freqs, outp)

        # Show the user
        plt.show()

def _preproc_producer_fn(q, root, sample_rate, nchannels, bytewidth, dice_to_seconds, nworkers, fraction_to_preprocess):
    """
    Function to act as the target for the producer process.

    The producer process crawls through `root` recursively, looking for WAV files.
    Whenever it finds a WAV file, it loads the entire thing into memory, resamples it appropriately,
    dices it into pieces of `dice_to_seconds` seconds, then puts each segment into `q`.

    :param q: The multiprocessing.Queue to use to communicate with the workers.
    :param root: The root directory to start looking for WAV files from.
    :param sample_rate: The Hz to resample any audio files it encounters to.
    :param nchannels: The number of audio channels to resample to.
    :param bytewidth: The byte width to resample to.
    :param dice_to_seconds: The number of seconds per slice of audio. Not all audio segments are guarenteed to be this length.
    :param nworkers: The number of worker threads.
    :param fraction_to_preprocess: The probability we will actually use a file. Useful for debugging - so that we don't have to preprocess everything.
    """
    assert os.path.isdir(root), "{} is not a valid path.".format(root)

    # Cache all the wav files
    fpathcache = set()
    for dirpath, _subdirs, fpaths in os.walk(root):
        for fpath in fpaths:
            if fpath.lower().endswith(".wav"):
                fpathcache.add(os.path.join(dirpath, fpath))

    # Now walk through the cache and deal with it. This allows us to tell how many items there are.
    for fpath in tqdm.tqdm(fpathcache):
        # Only take each fpath with some probability
        if random.random() < fraction_to_preprocess:
            # Dice it up, resample it, and put the pieces on the queue
            try:
                logging.debug("Attempting to preprocess {}".format(fpath))
                master_segment = audiosegment.from_file(fpath)
            except Exception as e:
                logging.warn("Could not load {} into an AudioSegment object, reason: {}".format(fpath, e))
                continue
            master_segment = master_segment.resample(sample_rate_Hz=sample_rate, sample_width=bytewidth, channels=nchannels)
            pieces = master_segment.dice(dice_to_seconds)
            for p in pieces:
                q.put(p)

    # We are done with all the data, so let's put the kill messages on the queue
    for _ in range(0, nworkers):
        q.put("DONE")

def _preproc_worker_fn(q, destination_folder, baby_detector_kwargs, language_detector_kwargs, baby_matrix, baby_model_stats, baby_raw_yes,
                            baby_event_length_s, language_matrix, language_model_stats, language_event_length_s):
    """
    Function to act as the target for the worker processes.

    The workers take items off the Queue and do the entire preprocessing pipeline to them.
    Then they save them in `destination_folder`.

    See the calling and the config file for an explanation of each of the parameters.
    """
    # Monotonically increasing ids for the different languages' AudioSegments that we save after preprocessing
    chinese_counter = 0
    english_counter = 0

    while True:
        next_q_item = q.get(block=True)
        if next_q_item == "DONE":
            return
        else:
            # Next q item should be an audio segment

            try:
                # -- Remove silence --
                # The required duration of silence for removal eligibility
                silence_duration_s = 5.0

                # If the segment is not as long as the amount of silence necessary to be eligible for trimming,
                # we should give up on this segment
                if next_q_item.duration_seconds <= silence_duration_s:
                    continue

                segment = next_q_item.filter_silence(duration_s=silence_duration_s, threshold_percentage=5.0)

                # If we only have a little bit of sound left after silence removal, we should give up on it
                if segment.duration_seconds < 1.0:
                    continue

                # Dice to 20 seconds each
                segments = segment.dice(20)

                for segment in segments:
                    ### TODO: The below commented out code is a refinement to the preprocessing pipeline such that we can strip baby sounds and remove
                    ###       determine which language is being used. But the models for these are not trained yet, and since training them is non-trivial,
                    ###       (though almost all of the infrustructure is in place to do so - it would just be non-trivial to give the models the attention
                    ###       they deserve), I will put off doing this for now and implement it if I have time (based on priorities).
                    chinese = None
                    english = segment
                    ## -- Remove baby --
                    #baby_detector = vd.VoiceDetector(**baby_detector_kwargs)
                    #events = segment.detect_event(baby_detector, baby_detector_kwargs['ms'], baby_matrix, baby_model_stats, baby_event_length_s)
                    #negatives = [tup[1] for tup in events if tup[0] == 'n']
                    #if len(negatives) == 0:
                    #    continue  # This segment was all baby all the time
                    #elif len(negatives) == 1:
                    #    segment = negatives[0]
                    #else:
                    #    segment = negatives[0].reduce(segment[1:])

                    ## If we only have a little bit of sound left after baby removal, we should give up on it
                    #if segment.duration_seconds < 1.0:
                    #    continue

                    ## -- Determine language --
                    #language_detector = vd.VoiceDetector(**language_detector_kwargs)
                    #events = segment.detect_event(language_detector, language_detector_kwargs['ms'], language_matrix, language_model_stats, language_event_length_s)
                    #chinese = [tup[1] for tup in events if tup[0] == 'y']  # TODO: Assumes we are using a Chinese detector model rather than an English detector
                    #english = [tup[1] for tup in events if tup[0] == 'n']
                    #if len(chinese) == 0:
                    #    chinese = None
                    #elif len(chinese) == 1:
                    #    chinese = chinese[0]
                    #else:
                    #    chinese = chinese[0].reduce(chinese[1:])

                    #if len(english) == 0:
                    #    english = None
                    #elif len(english) == 1:
                    #    english = english[0]
                    #else:
                    #    english = english[0].reduce(english[1:])

                    # -- Save to appropriate file with label --
                    if chinese is not None:
                        chinese.export("{}/chinese_{}.wav".format(destination_folder, chinese_counter), format="WAV")
                        chinese_counter += 1

                    if english is not None:
                        english.export("{}/english_{}.wav".format(destination_folder, english_counter), format="WAV")
                        english_counter += 1
            except Exception as e:
                logging.debug("Problem with an audio segment: {}".format(e))

def _run_preprocessing_pipeline(config):
    """
    Preprocesses the data according to config's properties.
    """
    logging.info("Preprocessing...")

    # This is the folder we will get stuff from
    root_folder = config.getstr('preprocessing', 'root')

    # This is the folder we will put stuff in after we are done preprocessing
    destination_folder = config.getstr('preprocessing', 'destination')

    # This is the fraction of the files in the root that we will actually bother to preprocess - others are ignored. Useful for testing.
    fraction_to_preprocess = config.getfloat('preprocessing', 'fraction_to_preprocess')

    # This is the sample rate of the audio - we resample to this (Hz)
    sample_rate = config.getfloat('preprocessing', 'sample_rate_hz')

    # This is the number of channels we will resample the audio to
    nchannels = config.getint('preprocessing', 'nchannels')

    # This is the bytewidth we will resample the audio to
    bytewidth = config.getint('preprocessing', 'bytewidth')

    # This is the number of seconds that the producer will slice the raw audio files into for each consumer to process
    dice_to_seconds = config.getfloat('preprocessing', 'dice_to_seconds')

    # These are the configurations for the baby detector
    baby_detector_kwargs = {}
    baby_detector_kwargs['sample_rate_hz'] = config.getfloat('preprocessing', 'baby_detector_sample_rate_hz')
    baby_detector_kwargs['sample_width_bytes'] = config.getint('preprocessing', 'baby_detector_sample_width_bytes')
    baby_detector_kwargs['ms'] = config.getfloat('preprocessing', 'baby_detector_ms')
    baby_detector_kwargs['model_type'] = config.getstr('preprocessing', 'baby_detector_model_type')
    baby_detector_kwargs['window_length_ms'] = config.getfloat('preprocessing', 'baby_detector_window_length_ms')
    baby_detector_kwargs['overlap'] = config.getfloat('preprocessing', 'baby_detector_overlap')
    baby_detector_kwargs['spectrogram_shape'] = config.getlist('preprocessing', 'baby_detector_spectrogram_shape')

    # These are the configurations for the English/Chinese detector
    language_detector_kwargs = {}
    language_detector_kwargs['sample_rate_hz'] = config.getfloat('preprocessing', 'language_detector_sample_rate_hz')
    language_detector_kwargs['sample_width_bytes'] = config.getint('preprocessing', 'language_detector_sample_width_bytes')
    language_detector_kwargs['ms'] = config.getfloat('preprocessing', 'language_detector_ms')
    language_detector_kwargs['model_type'] = config.getstr('preprocessing', 'language_detector_model_type')
    language_detector_kwargs['window_length_ms'] = config.getfloat('preprocessing', 'language_detector_window_length_ms')
    language_detector_kwargs['overlap'] = config.getfloat('preprocessing', 'language_detector_overlap')
    language_detector_kwargs['spectrogram_shape'] = config.getlist('preprocessing', 'language_detector_spectrogram_shape')

    # These are the event detection parameters for the baby noises
    baby_p_yes_to_no = config.getfloat('preprocessing', 'baby_detector_p_yes_to_no')
    baby_p_no_to_yes = config.getfloat('preprocessing', 'baby_detector_p_no_to_yes')
    baby_positive_predictive_value = config.getfloat('preprocessing', 'baby_detector_positive_predictive_value')
    baby_negative_predictive_value = config.getfloat('preprocessing', 'baby_detector_negative_predictive_value')
    baby_event_length_s = config.getfloat('preprocessing', 'baby_detector_event_length_s')
    baby_raw_yes = config.getfloat('preprocessing', 'baby_detector_raw_yes')
    baby_matrix = [baby_p_yes_to_no, baby_p_no_to_yes]
    baby_model_stats = [baby_positive_predictive_value, baby_negative_predictive_value]

    # These are the event detection parameters for the language detector
    language_p_yes_to_no = config.getfloat('preprocessing', 'language_detector_p_yes_to_no')
    language_p_no_to_yes = config.getfloat('preprocessing', 'language_detector_p_no_to_yes')
    language_positive_predictive_value = config.getfloat('preprocessing', 'language_detector_positive_predictive_value')
    language_negative_predictive_value = config.getfloat('preprocessing', 'language_detector_negative_predictive_value')
    language_event_length_s = config.getfloat('preprocessing', 'language_detector_event_length_s')
    language_matrix = [language_p_yes_to_no, language_p_no_to_yes]
    language_model_stats = [language_positive_predictive_value, language_negative_predictive_value]

    # Log some stuff
    logging.info("Will get data from: {}".format(root_folder))
    logging.info("Will put data in: {}".format(destination_folder))
    logging.info("Parameters for baby detector: {}".format(baby_detector_kwargs))
    logging.info("Parameters for langauge detector: {}".format(language_detector_kwargs))

    # Make a queue
    q = mp.Queue()

    # Number of total consumers
    nworkers = config.getint('preprocessing', 'nworkers')

    # Make a process that crawls the root directory looking for WAVs
    producer = mp.Process(target=_preproc_producer_fn, daemon=True, args=(q, root_folder, sample_rate, nchannels, bytewidth, dice_to_seconds, nworkers, fraction_to_preprocess))
    producer.start()

    # Make a pool of processes that each sit around waiting for segments of audio on the queue
    worker_args = (q, destination_folder, baby_detector_kwargs, language_detector_kwargs, baby_matrix, baby_model_stats, baby_raw_yes,
                        baby_event_length_s, language_matrix, language_model_stats, language_event_length_s)
    consumers = [mp.Process(target=_preproc_worker_fn, daemon=True, args=worker_args, name="preproc_worker_{}".format(i)) for i in range(nworkers)]
    for c in consumers:
        c.start()

    # Wait until everyone is finished
    producer.join()
    for c in consumers:
        c.join()

def _convert_to_images(config):
    """
    Converts all the preprocessed audio files into spectrograms saved as PNG files.
    """
    logging.info("Preprocessing into images...")

    # Get a bunch of config stuff
    folder_to_convert_from = config.getstr('preprocessing', 'destination')
    folder_to_save_images = config.getstr('preprocessing', 'images_destination')
    folder_to_save_wavs = config.getstr('preprocessing', 'folder_to_save_wavs')
    seconds_per_spectrogram = config.getfloat('preprocessing', 'seconds_per_spectrogram')
    window_length_s = config.getfloat('preprocessing', 'spectrogram_window_length_s')
    overlap = config.getfloat('preprocessing', 'spectrogram_window_overlap')
    fraction_to_preprocess = config.getfloat('preprocessing', 'fraction_to_preprocess')
    resample_to_hz = config.getfloat('preprocessing', 'spectrogram_sample_rate_hz')
    use_filterbank = config.getbool('preprocessing', 'use_filterbank')

    # Cache all the wav files
    fpathcache = set()
    for dirpath, _subdirs, fpaths in os.walk(folder_to_convert_from):
        for fpath in fpaths:
            if fpath.lower().endswith(".wav"):
                fpathcache.add(os.path.join(dirpath, fpath))

    # Now walk through the cache and deal with it. This allows us to tell how many items there are.
    for fpath in tqdm.tqdm(fpathcache):
        # Only preprocess with some probability (for testing)
        if random.random() < fraction_to_preprocess:
            try:
                segment = audiosegment.from_file(fpath)
                segments = segment.dice(seconds_per_spectrogram)
                for idx, segment in enumerate(segments):
                    segment = segment.resample(sample_rate_Hz=resample_to_hz)
                    if use_filterbank:
                        # TODO: Apply a bank of filters, then recombine before taking the spectrogram
                        raise NotImplementedError("Filterbank is not currently implemented.")
                    _fs, _ts, amps = segment.spectrogram(window_length_s=window_length_s, overlap=overlap)
                    #amps = 10.0 * np.log10(amps + 1e-9)  # This seems to make the output array a little harder to see in black/white
                    amps *= 255.0 / np.max(np.abs(amps))
                    amps = amps.astype(np.uint8)
                    _, fname = os.path.split(fpath)
                    fname_to_save = "{}_{}.png".format(fname, idx)
                    path_to_save = os.path.join(folder_to_save_images, fname_to_save)
                    imageio.imwrite(path_to_save, amps)
                    # Also save the segment that we created the spectrogram from
                    segment.export("{}_{}.wav".format(os.path.join(folder_to_save_wavs, fname), idx), format="WAV")
            except Exception as e:
                logging.warn("Could not convert file {}: {}".format(fpath, e))

def _build_vae1(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds model 1 of the VAE.
    """
    if is_variational:
        # Encoder model
        inputs = Input(shape=input_shape, name="encoder-input")                 # (-1, 241, 20, 1)
        x = Conv2D(128, (8, 2), strides=(2, 1), activation='relu', padding='valid')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 2), strides=(2, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (8, 2), strides=(2, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (9, 2), strides=(2, 2), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(8, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
        x = Flatten()(x)
        encoder = Dense(128, activation='relu')(x)

        # Decoder model
        decoderinputs = Input(shape=(latent_dim,), name='decoder-input')
        x = Dense(128, activation='relu')(decoderinputs)
        x = Reshape(target_shape=(4, 4, 8))(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(8, (3 ,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3 ,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3 ,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3 ,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 4), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((1, 2))(x)
        x = Conv2D(32, (8, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        decoder = Conv2D(1, (2, 1), activation='relu', padding='valid')(x)

        autoencoder = vae.VariationalAutoEncoder(input_shape, latent_dim, optimizer, loss,
                                                    kl_loss_prop=kl_loss_prop, recon_loss_prop=recon_loss_prop, std_loss_prop=std_loss_prop,
                                                    encoder=encoder, decoder=decoder, inputlayer=inputs,
                                                    decoderinputlayer=decoderinputs, tbdir=tbdir)
    else:
        inputs = Input(shape=input_shape, name="encoder-input")                 # (-1, 241, 20, 1)
        x = Conv2D(128, (8, 2), strides=(2, 1), activation='relu', padding='valid')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 2), strides=(2, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (8, 2), strides=(2, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (9, 2), strides=(2, 2), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(8, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
        x = Flatten()(x)
        encoder = Dense(128, activation='relu')(x)

        decoderinputs = Input(shape=(latent_dim,), name='decoder-input')
        x = Dense(128, activation='relu')(decoderinputs)
        x = Reshape(target_shape=(4, 4, 8))(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(8, (3 ,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3 ,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3 ,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3 ,3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 4), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((1, 2))(x)
        x = Conv2D(32, (8, 2), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2, 2))(x)
        decoder = Conv2D(1, (2, 1), activation='relu', padding='valid')(x)

        autoencoder = vanilla.AutoEncoder(input_shape, latent_dim, optimizer, loss, encoder, decoder, inputs, decoderinputs)

    return autoencoder

def _build_vae2(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds model 2 of the VAE.
    """
    if is_variational:
        # Encoder model
        inputs = Input(shape=input_shape, name="encoder-input")                 # (-1, 81, 18, 1)
        x = Conv2D(128, (8, 3), strides=(2, 1), activation='relu', padding='valid')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (6, 3), strides=(1, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        encoder = Dense(128, activation='relu')(x)

        decoderinputs = Input(shape=(latent_dim,), name='decoder-input')
        x = Dense(128, activation='relu')(decoderinputs)
        x = Reshape(target_shape=(4, 4, 8))(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (6, 3), activation='relu', padding='valid')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 1), activation='relu', padding='valid')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (8, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        decoder = Conv2D(1, (2, 1), activation='relu', padding='valid')(x)

        autoencoder = vae.VariationalAutoEncoder(input_shape, latent_dim, optimizer, loss,
                                                    kl_loss_prop=kl_loss_prop, recon_loss_prop=recon_loss_prop, std_loss_prop=std_loss_prop,
                                                    encoder=encoder, decoder=decoder, inputlayer=inputs,
                                                    decoderinputlayer=decoderinputs, tbdir=tbdir)
    else:
        inputs = Input(shape=input_shape, name="encoder-input")                 # (-1, 81, 18, 1)
        x = Conv2D(128, (8, 3), strides=(2, 1), activation='relu', padding='valid')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(64, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (6, 3), strides=(1, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        encoder = Dense(128, activation='relu')(x)

        decoderinputs = Input(shape=(latent_dim,), name='decoder-input')
        x = Dense(128, activation='relu')(decoderinputs)
        x = Reshape(target_shape=(4, 4, 8))(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (6, 3), activation='relu', padding='valid')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
        x = UpSampling2D((2, 2))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 1), activation='relu', padding='valid')(x)
        x = UpSampling2D((2, 1))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (8, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (8, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, (3, 1), activation='relu', padding='valid')(x)
        x = BatchNormalization()(x)
        decoder = Conv2D(1, (2, 1), activation='relu', padding='valid')(x)

        autoencoder = vanilla.AutoEncoder(input_shape, latent_dim, optimizer, loss, encoder, decoder, inputs, decoderinputs)

    return autoencoder

def _build_vae(config):
    """
    Builds the Variational AutoEncoder and returns it. Uses parameters from the config file.
    """
    # Get the input shape that the Encoder layer expects
    input_shape = config.getlist('autoencoder', 'input_shape', type=int)

    # Get the dimensionality of the embedding space
    latent_dim = config.getint('autoencoder', 'nembedding_dims')

    # Get the optimizer
    optimizer = config.getstr('autoencoder', 'optimizer')

    # Get the reconstructive loss function
    loss = config.getstr('autoencoder', 'loss')

    # Value between 0 and 1.0 that shows how much of the whole VAE loss function to assign to KL loss
    kl_loss_proportion = config.getfloat('autoencoder', 'kl_loss_proportion')

    # Value between 0 and 1.0 that shows how much of the whole VAE loss function to assign to reconstructive loss
    reconstructive_loss_proportion = config.getfloat('autoencoder', 'reconstructive_loss_proportion')

    # Value between 0 and 1.0 that shows how much of the whole VAE loss function to assign to the variance portion
    std_loss_proportion = config.getfloat('autoencoder', 'std_loss_proportion')

    # Are we variational or vanilla?
    is_variational = config.getbool('autoencoder', 'is_variational')

    # Get TensorBoard directory
    tbdir = config.getstr('autoencoder', 'tbdir')
    assert os.path.isdir(tbdir) or tbdir.lower() == "none", "{} is not a valid directory. Please fix tbdir in 'autoencoder' section of config file.".format(tbdir)
    # Now create a subdirectory in it
    experiment_name = config.getstr('experiment', 'name')
    tbdir = os.path.join(tbdir, experiment_name)
    os.makedirs(tbdir, exist_ok=True)

    # Remove anything in the tbdir already
    if tbdir is not None:
        shutil.rmtree(tbdir)  # Remove the directory and everything in it
        os.mkdir(tbdir)       # Remake the directory

    if list(input_shape) == [241, 20, 1]:
        return _build_vae1(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_proportion, reconstructive_loss_proportion, std_loss_proportion)
    elif list(input_shape) == [81, 18, 1]:
        return _build_vae2(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_proportion, reconstructive_loss_proportion, std_loss_proportion)
    else:
        raise ValueError("Spectrogram shape must be one of the allowed input shapes for the different VAE models, but is {}".format(input_shape))

def _train_vae(autoencoder, config):
    """
    Train the given `autoencoder` according to parameters listed in `config`.
    """
    # The root of the preprocessed data directory
    root = config.getstr('autoencoder', 'preprocessed_data_root')
    assert os.path.isdir(root), "{} is not a valid path.".format(root)

    # The root of the test split
    test = config.getstr('autoencoder', 'testsplit_root')
    if test.endswith("useless_subdirectory") or test.endswith("useless_subdirectory" + os.sep):
        test = os.path.dirname(test)
    assert os.path.isdir(test), "{} is not a valid path.".format(test)

    # Get whether or not we should visualize during training
    visualize = config.getbool('autoencoder', 'visualize')

    # The number of spectrograms per batch
    batchsize = config.getint('autoencoder', 'batchsize')

    # The number of workers to help with the data collection and feeding
    nworkers = config.getint('autoencoder', 'nworkers')

    # The number of epochs to train. Note that we define how long an epoch is manually
    nepochs = config.getint('autoencoder', 'nepochs')

    # The number of steps we have in an epoch
    steps_per_epoch = config.getint('autoencoder', 'steps_per_epoch')

    # The shape of the images
    imshapes = config.getlist('autoencoder', 'input_shape')[0:2]  # take only the first two dimensions (not channels)
    imshapes = [int(i) for i in imshapes]

    # Deal with the visualization stuff if we are visualizing
    if visualize:
        vaeviz = VaeVisualizer()
        model = autoencoder._vae
        fetches = [tf.assign(vaeviz.var_y_true, model.targets[0], validate_shape=False),
                tf.assign(vaeviz.var_y_pred, model.outputs[0], validate_shape=False),
                tf.assign(vaeviz.var_x, model.inputs[0], validate_shape=False)]
        model._function_kwargs = {'fetches': fetches}
        callbacks = [vaeviz]
    else:
        callbacks = []

    logging.info("Loading images from {}".format(root))
    nsteps_per_validation = steps_per_epoch
    imreader = preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    print("Creating datagen...")
    datagen = imreader.flow_from_directory(root,
                                            target_size=imshapes,
                                            color_mode='grayscale',
                                            classes=None,
                                            class_mode='input',
                                            batch_size=batchsize,
                                            shuffle=True,
                                            save_to_dir=None,
                                            save_format='png')
    print("Creating testgen...")
    testreader = preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    testgen = testreader.flow_from_directory(test,#root,
                                            target_size=imshapes,
                                            color_mode='grayscale',
                                            classes=None,
                                            class_mode='input',
                                            batch_size=batchsize,
                                            shuffle=True,
                                            save_to_dir=None,
                                            save_format='png')
    print("Training...")
    autoencoder.fit_generator(datagen,
                              batchsize,
                              epochs=nepochs,
                              save_models=True,
                              steps_per_epoch=steps_per_epoch,
                              use_multiprocessing=False,
                              workers=nworkers,
                              callbacks=callbacks,
                              validation_data=testgen,
                              validation_steps=nsteps_per_validation)

def _infer_with_vae(autoencoder: vae.VariationalAutoEncoder, config) -> [(str, str, np.array)]:
    """
    Returns a list of tuples of the form (spectrogram_fpath, audiofile_fpath, embedding coordinates as NP array).
    """
    if autoencoder is None:
        raise NotImplementedError("Currently, we need an autoencoder here.")

    try:
        # Try interpreting this configuration as an int - if so, that's the number of random
        # targets drawn from the test split
        print("Finding all the images in the test split...")
        testdir = config.getstr('autoencoder', 'testsplit_root')
        pathnames = [p for p in os.listdir(testdir) if os.path.splitext(p)[1].lower() == ".png"]
        paths = [os.path.abspath(os.path.join(testdir, p)) for p in pathnames]

        # Draw n random items from the test split as our targets
        nitems_to_mimic = config.getint('synthesizer', 'mimicry-targets')
        mimicry_targets = [random.choice(paths) for _ in range(nitems_to_mimic)]
    except ConfigError:
        # Other possibility is that the configuration item is a list of file paths to target
        mimicry_targets = config.getlist('synthesizer', 'mimicry-targets')

    print("Reading in all the images...")
    specs = [imageio.imread(p) / 255.0 for p in mimicry_targets]
    specs = np.expand_dims(np.array(specs), -1)

    logging.info("Found {} spectrograms to feed into the autoencoder at inference time.".format(specs.shape[0]))

    print("Predicting on each image in the mimicry list...")
    if isinstance(autoencoder, vae.VariationalAutoEncoder):
        # The output of the encoder portion of the model is three items: Mean, LogVariance, and Value sampled from described distribution
        # We will only use the means of the distros, not the actual encodings, as the means are simply
        # what is approximated by the encodings (though with uncertainty - the amount of uncertainty is measured in _logvars)
        means, _logvars, _encodings = autoencoder._encoder.predict(specs)
    else:
        # The encoder only outputs embeddings. But these are functionally the same as means for our purposes.
        means = autoencoder._encoder.predict(specs)

    # We have a list of .png files. But we want the WAVs that they correspond to. Let's find them.
    pngs_and_means = [tup for tup in zip(mimicry_targets, means)]
    folder = config.getstr('preprocessing', 'folder_to_save_wavs')  # This is where we saved the corresponding wav files
    fpaths_and_means = [(p, ae.convert_spectpath_to_audiofpath(folder, p), embedding) for p, embedding in pngs_and_means]
    if len(fpaths_and_means) < 10:
        logging.info("mimicry targets:\n{}".format(fpaths_and_means))
    else:
        logging.info("Too many mimicry targets to list. Have: {}".format(len(fpaths_and_means)))

    return fpaths_and_means

def _train_or_load_autoencoder(train_vae: bool, config) -> vae.VariationalAutoEncoder:
    """
    If `train_vae` is `True`, we use the config file to determine stuff and train a VAE.
    If it is `False`, we attempt to use the config to load an already trained VAE.
    If we cannot find one, we return None and issue a warning over logging.
    """
    # Build the right autoencoder model
    autoencoder = _build_vae(config)

    if train_vae:
        name = config.getstr('experiment', 'name')
        ae_savedir = config.getstr('autoencoder', 'weights_path')
        ae_savefpath = os.path.join(ae_savedir, name)
        fpath_to_save = "{}.h5".format(ae_savefpath)

        # Train the autoencoder
        logging.info("Training the autoencoder. Models will be saved to: {}".format(fpath_to_save))
        _train_vae(autoencoder, config)

        # Save the model's weights
        autoencoder.save_weights(fpath_to_save)
    else:
        try:
            # Load the weights into the constructed autoencoder model
            autoencoder_weights_fpath = config.getstr('autoencoder', 'weights_path')
            logging.info("Attempting to load autoencoder weights from {}".format(autoencoder_weights_fpath))
            autoencoder.load_weights(autoencoder_weights_fpath)
        except OSError:
            # Couldn't find a model and we weren't told to train one. Hopefully user knows what they're doing
            logging.warn("Could not find any autoencoder.")
            autoencoder = None

    return autoencoder

def run(config, savedir, preprocess=False, preprocess_part_two=False, pretrain_synth=False, train_vae=False, train_synth=False):
    """
    Entry point for Phase 1.

    Initializes and pretrains the voice synthesization network to vocalize;
    Creates and trains a variational autoencoder on the entire Oliver data set;
    Applies a clustering algorithm to the embeddings that the VAE comes up with once it is trained;
    Determines a prototype sound for each cluster;
    Finishes training the voice synthesizer to mimic these sounds based on which embedding it observes.

    If `preprocess` is True, we will preprocess all the data as part of the experiment. See the config file for details.
    If `preprocess_part_two` is True, we will convert all the preprocessed sound files into black and white images of spectrograms.
    If `pretrain_synth` is True, we will pretrain the voice synthesizer to make noise.
    If `train_vae` is True, we will train the variational autoencoder on the preprocessed data.
    If `train_synth` is True, we will train the voice synthesizer to mimic the prototypical proto phonemes.
    """
    # Potentially preprocess the audio
    if preprocess:
        print("Preprocessing all sounds. This will take close to forever...")
        _run_preprocessing_pipeline(config)

    # Convert the preprocessed audio files into spectrograms and save them as image files
    if preprocess_part_two:
        print("Converting all preprocessed sound files into spectrograms and saving them as images. This will take the rest of forever...")
        _convert_to_images(config)

    # Pretrain the voice synthesizer to make non-specific noise
    if pretrain_synth:
        print("Pretraining the voice synthesizer. Learning to coo...")
        synthmodel = motorcortex.SynthModel(config)
        synthmodel.pretrain()
        production.analyze_pretrained_model(config, synthmodel.phase0_artifacts_dir, savedir, "pretraining", synthmodel)
    else:
        synthmodel = None

    # Get a trained autoencoder
    autoencoder = _train_or_load_autoencoder(train_vae, config)
    if train_vae:
        print("Analyzing the autoencoder...")
        ae.analyze(config, autoencoder, savedir)

    # Train the motor cortex to produce sounds from these different embeddings
    # The synthesizer uses the autoencoder to evaluate how close its output is to the target sound
    # in latent space.
    if train_synth:
        # Now use the VAE on the test split and save pairs of (audiofile, coordinates in embedding space)
        mimicry_targets = _infer_with_vae(autoencoder, config)

        if synthmodel is None:
            synthmodel = motorcortex.SynthModel(config)

        trained_synth_models = motorcortex.train_on_targets(config, synthmodel, mimicry_targets, autoencoder)
        production.analyze_models(config, trained_synth_models, savedir, [target[1] for target in mimicry_targets])
