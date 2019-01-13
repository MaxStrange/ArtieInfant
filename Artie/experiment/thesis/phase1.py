"""
This is the phase 1 file.

This file's API consists simply of the function run(), which will run phase 1 of the thesis experiment.
"""
from internals.specifics import rl                      # pylint: disable=locally-disabled, import-error
from experiment import configuration                    # pylint: disable=locally-disabled, import-error
from senses.voice_detector import voice_detector as vd  # pylint: disable=locally-disabled, import-error
import audiosegment
import logging
import multiprocessing as mp
import os

def _preproc_producer_fn(q, root, sample_rate, nchannels, bytewidth, dice_to_seconds, nworkers):
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
    """
    # Walk through the data directory
    for dirpath, _subdirs, fpaths in os.walk(root):
        for fpath in fpaths:
            # For each WAV file
            if fpath.lower().endswith(".wav"):
                # Dice it up, resample it, and put it on the queue
                segment_fpath = os.path.join(dirpath, fpath)
                try:
                    master_segment = audiosegment.from_file(segment_fpath)
                except Exception as e:
                    logging.warn("Could not load {} into an AudioSegment object, reason: {}".format(segment_fpath, e))
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

            # -- Remove silence --
            # The required duration of silence for removal eligibility is 1/100th of the length of the slice,
            # but with a minimum of 0.5 seconds and a maximum of 1.0 seconds
            silence_duration_s = min(max(next_q_item.duration_seconds / 100.0, 1.0), 0.5)
            segment = next_q_item.filter_silence(duration_s=silence_duration_s, threshold_percentage=1)

            # If we only have a little bit of sound left after silence removal, we should give up on it
            if segment.duration_seconds < 1.0:
                continue

            # -- Remove non-voice --
            voiced_segments = [tup[1] for tup in segment.detect_voice() if tup[0] == 'v']
            if len(voiced_segments) == 0:
                continue  # This segment had no voice in it
            elif len(voiced_segments) == 1:
                voiced_segment = voiced_segments[0]
            else:
                voiced_segment = voiced_segments[0].reduce(voiced_segments[1:])

            # If we only have a little bit of sound left after voice detection, we should give up on it
            if voiced_segment.duration_seconds < 1.0:
                continue

            ### TODO: The below commented out code is a refinement to the preprocessing pipeline such that we can strip baby sounds and remove
            ###       determine which language is being used. But the models for these are not trained yet, and since training them is non-trivial,
            ###       (though almost all of the infrustructure is in place to do so - it would just be non-trivial to give the models the attention
            ###       they deserve), I will put off doing this for now and implement it if I have time (based on priorities).
            chinese = None
            english = voiced_segment
            ## -- Remove baby --
            #baby_detector = vd.VoiceDetector(**baby_detector_kwargs)
            #events = voiced_segment.detect_event(baby_detector, baby_detector_kwargs['ms'], baby_matrix, baby_model_stats, baby_event_length_s)
            #negatives = [tup[1] for tup in events if tup[0] == 'n']
            #if len(negatives) == 0:
            #    continue  # This segment was all baby all the time
            #elif len(negatives) == 1:
            #    segment_sans_baby = negatives[0]
            #else:
            #    segment_sans_baby = negatives[0].reduce(segment_sans_baby[1:])

            ## If we only have a little bit of sound left after baby removal, we should give up on it
            #if segment_sans_baby.duration_seconds < 1.0:
            #    continue

            ## -- Determine language --
            #language_detector = vd.VoiceDetector(**language_detector_kwargs)
            #events = segment_sans_baby.detect_event(language_detector, language_detector_kwargs['ms'], language_matrix, language_model_stats, language_event_length_s)
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

def _run_preprocessing_pipeline(config):
    """
    Preprocesses the data according to config's properties.
    """
    logging.info("Preprocessing...")

    # This is the folder we will get stuff from
    root_folder = config.getstr('preprocessing', 'root')

    # This is the folder we will put stuff in after we are done preprocessing
    destination_folder = config.getstr('preprocessing', 'destination')

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

    # Make a process that crawls the root directory looking for WAVs
    producer = mp.Process(target=_preproc_producer_fn, args=(q, root_folder, sample_rate, nchannels, bytewidth, dice_to_seconds))
    producer.start()

    # Make a pool of processes that each sit around waiting for segments of audio on the queue
    nworkers = config.getint('preprocessing', 'nworkers')
    worker_args = (q, destination_folder, baby_detector_kwargs, language_detector_kwargs, baby_matrix, baby_model_stats, baby_raw_yes,
                        baby_event_length_s, language_matrix, language_model_stats, language_event_length_s)
    consumers = [mp.Process(target=_preproc_worker_fn, args=worker_args, name="preproc_worker_{}".format(i)) for i in range(nworkers)]
    for c in consumers:
        c.start()

    # Wait until everyone is finished
    producer.join()
    for c in consumers:
        c.join()

def run(preprocess=False, test=False):
    """
    Entry point for Phase 1.

    Initializes and pretrains the voice synthesization network to vocalize;
    Creates and trains a variational autoencoder on the entire Oliver data set;
    Applies a clustering algorithm to the embeddings that the VAE comes up with once it is trained;
    Determines a prototype sound for each cluster;
    Finishes training the voice synthesizer to mimic these sounds based on which embedding it observes.

    If `test` is True, we will load the testthesis.cfg config file instead of the thesis config.
    If `preprocess` is True, we will preprocess all the data as part of the experiment. See the config file for details.
    """
    # Load the right experiment configuration
    configname = "testthesis" if test else "thesis"
    config = configuration.load(configname)

    # Potentially preprocess the audio
    if preprocess:
        _run_preprocessing_pipeline(config)

    # Pretrain the voice synthesizer to make non-specific noise
    weightpathbasename, actor, critic = rl.pretrain(config)

    # TODO:
    #   VAE - train then run over a suitable sample of audio to save enough embeddings for the prototypes/clustering
    #       # Train the VAE to a suitable level of accuracy
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
