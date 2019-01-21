"""
This is the phase 1 file.

This file's API consists simply of the function run(), which will run phase 1 of the thesis experiment.
"""
from internals.specifics import rl                      # pylint: disable=locally-disabled, import-error
from internals.vae import vae                           # pylint: disable=locally-disabled, import-error
from experiment import configuration                    # pylint: disable=locally-disabled, import-error
from senses.voice_detector import voice_detector as vd  # pylint: disable=locally-disabled, import-error
from senses.dataproviders import sequence as seq        # pylint: disable=locally-disabled, import-error

from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, MaxPooling2D, Flatten, Reshape

import audiosegment
import logging
import multiprocessing as mp
import os
import random
import tqdm

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

    print("Recursively walking {}. This may take some time...".format(root))
    # Cache all the wav files
    fpathcache = set()
    for dirpath, _subdirs, fpaths in os.walk(root):
        for fpath in fpaths:
            if fpath.lower().endswith(".wav"):
                fpathcache.add(os.path.join(dirpath, fpath))
    print("Done. Preprocessing has begun.")

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
                # The required duration of silence for removal eligibility is 1/100th of the length of the slice,
                # but with a minimum of x seconds and a maximum of y seconds
                silence_duration_s = 10.0#min(max(next_q_item.duration_seconds / 100.0, 5.0), 5.0)

                # If the segment is not as long as the amount of silence necessary to be eligible for trimming,
                # we should give up on this segment
                if next_q_item.duration_seconds <= silence_duration_s:
                    continue

                segment = next_q_item.filter_silence(duration_s=silence_duration_s, threshold_percentage=0.5)

                # If we only have a little bit of sound left after silence removal, we should give up on it
                if segment.duration_seconds < 1.0:
                    continue

                # -- Remove non-voice --
                voiced_segments = [tup[1] for tup in segment.detect_voice() if tup[0] == 'v']
                if len(voiced_segments) == 0:
                    continue  # This segment had no voice in it
                elif len(voiced_segments) == 1:
                    segment = voiced_segments[0]
                else:
                    segment = voiced_segments[0].reduce(voiced_segments[1:])

                # If we only have a little bit of sound left after voice detection, we should give up on it
                if segment.duration_seconds < 1.0:
                    continue

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
    root_folder = config.getstr('preprocessing', 'root')  # TODO: For some reason, getting a path from the config file results in a string that can't be interpreted as a path...

    # This is the folder we will put stuff in after we are done preprocessing
    destination_folder = config.getstr('preprocessing', 'destination')

    # TODO: Remove these two lines:
    root_folder = "/media/max/seagate8TB/thesis_audio/gold_data_do_not_modify"
    destination_folder = "/media/max/seagate8TB/thesis_audio/preprocessed_gold_data"

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

    # Encoder model
    inputs = Input(shape=input_shape, name="encoder_inputs")
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)           # (-1, 55, 19, 16)
    x = MaxPooling2D((2, 2), padding='same')(x)                                 # (-1, 28, 10, 16)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)                 # (-1, 28, 10, 8)
    x = MaxPooling2D((2, 2), padding='same')(x)                                 # (-1, 14, 5, 8)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)                 # (-1, 14, 5, 8)
    x = MaxPooling2D((2, 2), padding='same')(x)                                 # (-1, 7, 3, 8)
    x = Flatten()(x)                                                            # (-1, 168)
    encoder = Dense(32, activation='relu')(x)                                   # (-1, 32)

    # Decoder model
    intermediate_dim = (14, 5, 8)
    decoderinputs = Input(shape=(latent_dim,), name="decoder_inputs")
    x = Dense(np.product(intermediate_dim), activation='relu')(decoderinputs)   # (-1, 560)
    x = Reshape(target_shape=intermediate_dim)(x)                               # (-1, 14, 5, 8)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)                 # (-1, 14, 5, 8)
    x = UpSampling2D((2, 2))(x)                                                 # (-1, 28, 10, 8)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)                 # (-1, 28, 10, 8)
    x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)                # (-1, 28, 10, 16)
    x = UpSampling2D((2, 2))(x)                                                 # (-1, 56, 20, 16)
    decoder = Conv2D(1, (2, 2), activation='sigmoid', padding='valid')(x)       # (-1, 55, 19, 1)

    autoencoder = vae.VariationalAutoEncoder(input_shape, latent_dim, optimizer, loss, encoder, decoder, inputs, decoderinputs)
    return autoencoder

def _train_vae(autoencoder, config):
    """
    Train the given `autoencoder` according to parameters listed in `config`.
    """
    # TODO: Remove this line if you don't need the warning filter
    #warnings.simplefilter("ignore", ResourceWarning)

    # The root of the preprocessed data directory
    root = config.getstr('autoencoder', 'preprocessed_data_root')

    # The sample rate in Hz that the model expects
    sample_rate_hz = config.getfloat('autoencoder', 'sample_rate_hz')

    # The number of channels of audio the model expects
    nchannels = config.getint('autoencoder', 'nchannels')

    # The number of bytes per sample of the audio
    bytewidth = config.getint('autoencoder', 'bytewidth')

    # The total number of bytes in the preprocessed data directory
    total_bytes = 0 # TODO: Get this from the directory

    # Since WAV is uncompressed, we can get a fair approximation of the total sound duration
    # in the directory by doing some simple math
    ms_of_dataset = ((total_bytes / bytewidth) / sample_rate_hz) * 1000

    # The total ms per spectrogram
    ms = config.getfloat('autoencoder', 'ms')

    # The number of spectrograms per batch size
    batchsize = config.getint('autoencoder', 'batchsize')

    # The number of ms per batch
    ms_per_batch = ms * batchsize

    # The number of workers to help with the data collection and feeding
    nworkers = config.getint('autoencoder', 'nworkers')

    # The number of epochs to train. Note that we define how long an epoch is manually
    nepochs = config.getint('autoencoder', 'nepochs')

    # The number of steps we have in an epoch
    steps_per_epoch = config.getint('autoencoder', 'steps_per_epoch')

    # These args are passed into generate_n_spectrogram_batches(): (num batches to yield, batchsize, ms, label function)
    args = (None, batchsize, ms, None)

    # These are the keyword arguments to pass into generate_n_spectrogram_batches
    kwargs = {
        "normalize": True,
        "forever": True,
        "window_length_ms": None,
        "overlap": 0.5,
        "expand_dims": True,
    }
    sequence = seq.Sequence(ms_of_dataset,
                            ms_per_batch,
                            nworkers,
                            root,
                            sample_rate_hz,
                            nchannels,
                            bytewidth,
                            "generate_n_spectrogram_batches",
                            *args,
                            **kwargs)

    autoencoder.fit_generator(sequence,
                              batchsize,
                              epochs=nepochs,
                              save_models=True,
                              steps_per_epoch=steps_per_epoch,
                              use_multiprocessing=False,
                              workers=1)

def run(preprocess=False, test=False, pretrain_synth=False, train_vae=False, train_synth=False):
    """
    Entry point for Phase 1.

    Initializes and pretrains the voice synthesization network to vocalize;
    Creates and trains a variational autoencoder on the entire Oliver data set;
    Applies a clustering algorithm to the embeddings that the VAE comes up with once it is trained;
    Determines a prototype sound for each cluster;
    Finishes training the voice synthesizer to mimic these sounds based on which embedding it observes.

    If `test` is True, we will load the testthesis.cfg config file instead of the thesis config.
    If `preprocess` is True, we will preprocess all the data as part of the experiment. See the config file for details.
    If `pretrain_synth` is True, we will pretrain the voice synthesizer to make noise.
    If `train_vae` is True, we will train the variational autoencoder on the preprocessed data.
    If `train_synth` is True, we will train the voice synthesizer to mimic the prototypical proto phonemes.
    """
    # Load the right experiment configuration
    configname = "testthesis" if test else "thesis"
    config = configuration.load(configname)

    # Potentially preprocess the audio
    if preprocess:
        _run_preprocessing_pipeline(config)

    # Pretrain the voice synthesizer to make non-specific noise
    if pretrain_synth:
        weightpathbasename, actor, critic = rl.pretrain(config)

    # -- VAE -- train then run over a suitable sample of audio to save enough embeddings for the prototypes/clustering
    # Train the VAE to a suitable level of accuracy
    autoencoder = _build_vae(config)
    autoencoder_weights_fpath = config.getstr('autoencoder', 'weights_path')
    if train_vae:
        _train_vae(autoencoder, config)
        autoencoder.save_weights(autoencoder_weights_fpath)
    else:
        autoencoder.load_weights(autoencoder_weights_fpath)

    # TODO:
    #       # Use the trained VAE on ~1,000 (or more?) audio samples, saving each audio sample along with its embedding.
    #   Mean Shift Cluster - cluster the saved embeddings using sklearn.mean_shift_cluster (or whatever it's called). This will tell us how many clusters.
    #       # Load the saved embeddings into a dataset
    #       # Run the SKLearn algorithm over the dataset to determine how many clusters and to get cluster indexes.
    #   Determine prototypes - Go through and send each embedding into the clusterer to get its cluster index. Take one from each cluster to form a prototype.
    #       # Determine each saved embedding's cluster index.
    #       # Take 'quintessential' embeddings from each cluster and save them as prototypes.
    #   Finish training rl agent - Set up the environment with these prototypes and the weights of the pretrained agent. Train until it can mimic the prototypes given a cluster index.
    #       # Train the RL agent to mimic the given prototypes
    #   Inference: Save the trained RL agent and then you can use it to make noises based on cluster index inputs.
    #   You now have trained this:
    #   [1]  ->  /a/
    #   [2]  ->  /g/
    #   [.]  ->  ..
    #   [9]  ->  /b/
    #   [.]  ->  ..
    #   Which means that you now have a map of phonemes.
    #   You also have an embedding space that can be sampled from. That sample could then be run through the clusterer to determine
    #   the index of the sample, which would then determine which sound it was. I'm not sure what this gives you... but it seems like it might be important.

    # Clean up the weights
    os.remove(weightpathbasename + "_actor" + ".hdf5")
    os.remove(weightpathbasename + "_critic" + ".hdf5")
