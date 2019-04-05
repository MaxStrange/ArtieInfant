"""
This is the phase 1 file.

This file's API consists simply of the function run(), which will run phase 1 of the thesis experiment.
"""
from internals.motorcortex import motorcortex           # pylint: disable=locally-disabled, import-error
from internals.vae import vae                           # pylint: disable=locally-disabled, import-error
from senses.voice_detector import voice_detector as vd  # pylint: disable=locally-disabled, import-error
from senses.dataproviders import sequence as seq        # pylint: disable=locally-disabled, import-error

from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, MaxPooling2D, Flatten, Reshape, BatchNormalization
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


def _build_vae1(input_shape, latent_dim, optimizer, loss, tbdir):
    """
    Builds model 1 of the VAE.
    """
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
                                                encoder=encoder, decoder=decoder, inputlayer=inputs,
                                                decoderinputlayer=decoderinputs, tbdir=tbdir)
    return autoencoder

def _build_vae2(input_shape, latent_dim, optimizer, loss, tbdir):
    """
    Builds model 2 of the VAE.
    """
    raise NotImplementedError("This VAE architecture is not yet implemented.")

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

    # Get the loss function
    loss = config.getstr('autoencoder', 'loss')

    # Get TensorBoard directory
    tbdir = config.getstr('autoencoder', 'tbdir')
    assert os.path.isdir(tbdir) or tbdir.lower() == "none", "{} is not a valid directory. Please fix tbdir in 'autoencoder' section of config file.".format(tbdir)

    # Remove anything in the tbdir already
    if tbdir is not None:
        shutil.rmtree(tbdir)  # Remove the directory and everything in it
        os.mkdir(tbdir)       # Remake the directory

    if list(input_shape) == [241, 20, 1]:
        return _build_vae1(input_shape, latent_dim, optimizer, loss, tbdir)
    elif list(input_shape) == [161, 6, 1]:
        return _build_vae2(input_shape, latent_dim, optimizer, loss, tbdir)
    else:
        raise ValueError("Spectrogram shape must be one of the allowed input shapes for the different VAE models, but is {}".format(input_shape))

def _train_vae(autoencoder, config):
    """
    Train the given `autoencoder` according to parameters listed in `config`.
    """
    # The root of the preprocessed data directory
    root = config.getstr('autoencoder', 'preprocessed_data_root')
    assert os.path.isdir(root), "{} is not a valid path.".format(root)

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
    validation_fraction = 0.1
    nsteps_per_validation = steps_per_epoch * validation_fraction
    imreader = preprocessing.image.ImageDataGenerator(rescale=1.0/255.0, validation_split=validation_fraction)
    print("Creating datagen...")
    datagen = imreader.flow_from_directory(root,
                                            target_size=imshapes,
                                            color_mode='grayscale',
                                            classes=None,
                                            class_mode='input',
                                            batch_size=batchsize,
                                            shuffle=True,
                                            save_to_dir=None,
                                            save_format='png',
                                            subset="training")
    print("Creating testgen...")
    testgen = imreader.flow_from_directory(root,
                                            target_size=imshapes,
                                            color_mode='grayscale',
                                            classes=None,
                                            class_mode='input',
                                            batch_size=batchsize,
                                            shuffle=True,
                                            save_to_dir=None,
                                            save_format='png',
                                            subset="validation")
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

def _infer_with_vae(autoencoder: vae.VariationalAutoEncoder, config) -> [(str, np.array)]:
    """
    Uses the test split as found in the config file to test the autoencoder, and saves
    its embeddings along with the paths to the targets in a list that is returned.

    Returns a list of tuples of the form (audiofile, embedding coordinates as NP array).
    """
    testdir = config.getstr('autoencoder', 'testsplit_root')

    # Load a bunch of spectrograms into a batch
    assert os.path.isdir(testdir), "'testsplit_root' in 'autoencoder' must be a valid directory, but is {}".format(testdir)

    pathnames = [p for p in os.listdir(testdir) if os.path.splitext(p)[1].lower() == ".png"]
    paths = [os.path.abspath(os.path.join(testdir, p)) for p in pathnames]
    specs = [imageio.imread(p) / 255.0 for p in paths]
    specs = np.expand_dims(np.array(specs), -1)

    logging.info("Found {} spectrograms to feed into the autoencoder at inference time.".format(specs.shape[0]))

    # The output of the encoder portion of the model is three items: Mean, LogVariance, and Value sampled from described distribution
    # We will only use the means of the distros, not the actual encodings, as the means are simply
    # what is approximated by the encodings (though with uncertainty - the amount of uncertainty is measured in _logvars)
    means, _logvars, _encodings = autoencoder._encoder.predict(specs)

    return [tup for tup in zip(paths, means)]

def run(config, preprocess=False, preprocess_part_two=False, pretrain_synth=False, train_vae=False, train_synth=False):
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
        print("Converting all preprocessed sound files into spectrograms and saving them as images...")
        _convert_to_images(config)

    # Pretrain the voice synthesizer to make non-specific noise
    synthmodel = motorcortex.SynthModel(config)
    if pretrain_synth:
        print("Pretraining the voice synthesizer. Learning to coo...")
        synthmodel.pretrain()

    # Train the VAE to a suitable level of accuracy
    autoencoder = _build_vae(config)
    autoencoder_weights_fpath = config.getstr('autoencoder', 'weights_path')
    if train_vae:
        print("Training the variational autoencoder to embed the spectrograms into a two dimensional probabilistic embedding...")
        timestamp = datetime.datetime.now().strftime("date-%Y-%m-%d-time-%H-%M")
        fpath_to_save = "{}_{}.h5".format(autoencoder_weights_fpath, timestamp)
        logging.info("Training the autoencoder. Models will be saved to: {}".format(fpath_to_save))
        _train_vae(autoencoder, config)
        autoencoder.save_weights(fpath_to_save)
    else:
        logging.info("Attempting to load autoencoder weights from {}".format(autoencoder_weights_fpath))
        autoencoder.load_weights(autoencoder_weights_fpath)

    # Now use the VAE on the test split and save pairs of (audiofile, coordinates in embedding space)
    mimicry_targets = _infer_with_vae(autoencoder, config)

    # Train the motor cortex to produce sounds from these different embeddings
    # The synthesizer uses the autoencoder to evaluate how close its output is to the target sound
    # in latent space.
    if train_synth:
        # TODO: Fix train_on_targets to take whatever type mimicry_targets is
        # TODO: train_on_targets must take the autoencoder and use it in the genetic algorithm's loss function
        ret = motorcortex.train_on_targets(config, synthmodel, mimicry_targets, autoencoder)
        print("Got", ret, "back")  # TODO: Do something useful instead of just printing
