"""
This module provides a way to get the data from the raw directory and present it
as features for entry into the models. Its main API is the generate_data function.
"""
import audiosegment
import enum
import numpy as np
import os
import random
import src.utilities.utils as utils

class ClassLabels(enum.IntEnum):
    NO_VOICE = 0
    VOICE = 1

def _generate_segments(data_dir, shuffle=False, sampling_frequency_hz=32000,
                       sample_width=2, channels=1, ignore=None, include=None):
    """
    Generator function for creating a single audiosegment.AudioSegment object per wav file at a time
    from the raw data.

    :param data_dir: The top-level directory to walk down from, searching for wav files along the way.
    :param shuffle: Shuffle the data before yielding any.
    :param sampling_frequency_hz: Each wav file will be resampled to this frequency (if it isn't already at this one).
    :param sample_width: Each wav file will be resampled to this number of bytes per sample
                         (if it isn't already at this one).
    :param channels: Each wav file will be resampled to this number of channels (if it isn't already at this one).
    :param ignore: Directories to ignore.
    :param include: If not None, all the directories in here will be used, and no others.
    :returns: Yields a single wav file at a time, as a segment.
    """
    fpaths = []
    for root, __, wavpaths in os.walk(data_dir):
        if include:
            ignore_this_dir = not any([inc for inc in include if inc in root])
            if ignore_this_dir:
                continue
        elif ignore:
            ignore_this_dir = any([ig for ig in ignore if ig in root])
            if ignore_this_dir:
                continue

        for wavpath in wavpaths:
            if os.path.splitext(wavpath)[1].lower().endswith("wav"):
                fpaths.append(os.sep.join([root, wavpath]))

    if shuffle:
        random.shuffle(fpaths)
    for fpath in fpaths:
        seg = audiosegment.from_file(fpath)
        if seg.frame_rate != sampling_frequency_hz or seg.sample_width != sample_width or seg.channels != channels:
            seg = seg.resample(sample_rate_Hz=sampling_frequency_hz, sample_width=sample_width, channels=channels)
        yield seg, fpath

def calculate_steps_per_epoch(data_dir, samples_per_vector=5120, batch_size=64,
                              sampling_frequency_hz=32000, channels=1, ignore=None, include=None): #TODO: ignore and include
    """
    Figures out how many steps there are in an epoch. Simply number of samples in the dataset / batch_size.

    This will take a while to calculate, so once it is determined, it may make sense to store it and reuse it
    (assuming your data doesn't change between training runs).

    The calculation is this:
    steps_per_epoch = int(num_samples_in_dataset / samples_per_batch)

    Where:

    samples_per_batch = samples_per_vector * batch_size
    num_samples_in_dataset = SUM over all wav files { samples in wav file }
    samples_in_wav_file_i = len_in_seconds * sampling_frequency_hz * channels

    In order to calculate the number of steps, all of the following parameters must be known.

    :param data_dir: The top-level directory to walk down from, searching for wav files along the way.
    :param samples_per_vector: The number of samples in a single vector.
    :param batch_size: The number of vectors in a batch.
    :param sampling_frequency_hz: The frequency with which the wav files will be sampled.
    :param channels: The number of channels in a wav file.
    :param ignore: Directories to ignore.
    :param include: If not None, all the directories in here will be used, and no others.
    :returns: The number of vectors in the entire dataset.
    """
#    print("Calculating number of samples in dataset...")
#    num_samples_in_dataset = sum((len(seg) for seg, _path in _generate_segments(data_dir,
#                                                                                sampling_frequency_hz=sampling_frequency_hz,
#                                                                                channels=channels,
#                                                                                ignore=ignore,
#                                                                                include=include)))
#    utils.log("Number of samples calculated:", num_samples_in_dataset)
#    samples_per_batch = samples_per_vector * batch_size
#    utils.log("Samples per batch:", samples_per_batch)
#    steps_per_epoch = int(num_samples_in_dataset / samples_per_batch)
#    print("Steps per epoch:", steps_per_epoch)
    steps_per_epoch = 4050
    print("Using pre-cached value for steps_per_epoch. Uncomment the code in build_features.py if you need to recalculate.")
    print("  |-> steps_per_epoch:", steps_per_epoch)
    return steps_per_epoch

def generate_data(data_dir, samples_per_vector=5120, batch_size=64, sampling_frequency_hz=32000,
                    sample_width=2, channels=1, ignore=None, include=None):
    """
    Generates tuples of the form (input_vectors, class_label) indefinitely. Removes silence as it goes.

    The input_vectors that are generated by this function are simply samples_per_vector samples taken sequentially from
    wav files at random (without replacement until all wav files are used).

    So the general steps are this:
    - Get a wav file from the data repository along with its label.
    - Resample the file to the given properties (does not alter the original file).
    - Until the resampled file has run out of samples, read out samples_per_vector samples from it sequentially
      and pack them up into a vector. Do this batch_size number of times to create a batch.
    - If the wav file ran out before we got a full batch, start over from the beginning, but append the vectors
      to the batch we started.
    - Return the batch along with the labels.
    - In reality, we actually take some number of wav files at a time and yield FFT values from them sequentially
      but in random order between them, so that we have a list of say, five wav files that we are yielding from
      at a given time, at random, but in order of time. So that is to say, you don't know which file you are going
      to get a tuple from, but you do know that the next tuple you get, if it is from the same file, will be the
      next tuple in that file. The reason to do this is to increase the variance in each batch - otherwise we
      might get an entire batch that is just a single audio file - and thus a single label.

    :param data_dir: The top-level directory to walk down from, searching for wav files along the way.
    :param samples_per_vector: The size of the input_vector.
    :param batch_size: The number of vectors in a batch.
    :param sampling_frequency_hz: The raw wav file will be read in and resampled to this frequency.
    :param sample_width: The raw wav file will be read in and resampled using this number of bytes per sample.
    :param channels: The raw wav file will be read in and resampled to this number of channels.
    :param ignore: If not None, will skip any directories listed in this.
    :param include: If not None, will ONLY use these directories.
    :returns: Tuple of the form (input_vector, class_label)
    """
    ms_per_second = 1000
    seconds_per_sample = 1 / sampling_frequency_hz
    ms_per_vector = ms_per_second * seconds_per_sample * samples_per_vector
    if not ms_per_vector.is_integer():
        raise ValueError("1000 * 1/sampling_frequency * samples_per_vector must be a whole number. Got: "
                         + str(ms_per_vector))

    utils.log("Looping over dataset to provide data...")
    printed_a_batch = True  # Set to False to debug by looking at the features' FFTs
    batch_num = 0
    while True:
        # Keep generating forever
        try:
            vectors = []
            labels = []
            utils.log("Generating batch...")
            for i in range(batch_size):
                utils.log("Getting vector and label", i, increase_depth=1)
                vector, label = next(_generate_vector_by_vector(
                                        data_dir,
                                        sampling_frequency_hz,
                                        sample_width,
                                        channels,
                                        ms_per_vector,
                                        ignore=ignore,
                                        include=include))
                utils.log("Got a vector and label", increase_depth=1)
                vectors.append(vector)
                labels.append(label)
            if not printed_a_batch:
                utils.log("Printing the batch to csv files...")
                for i, (vector, label) in enumerate(zip(vectors, labels)):
                    fpath = "BATCH_" + str(batch_num) + "_" + str(label) + "_" + str(i) + ".csv"
                    with open(fpath, 'w') as f:
                        for val in vector:
                            f.write(str(val) + os.linesep)
                printed_a_batch = True
            utils.log("Yielding batch")
            vectors = np.array(vectors)
            batch = (vectors, labels)
            percent_no = 100 * len([l for l in labels if l == ClassLabels.NO_VOICE]) / len(labels)
            percent_yes = 100 - percent_no
            utils.log("")
            utils.log("Percent no/yes in this batch:", percent_no, "/", percent_yes)
            utils.log(batch)
            yield batch
            batch_num += 1
        except StopIteration:
            pass  # Just keep going

def _get_tups(segment, path_to_segment, ms_per_vector):
    """
    Generator to yield FFT values and labels from a segment. First filters the segment's silence.

    :param segment: The segment to yield from
    :param path_to_segment: The full path to the segment, including its name.
    :param ms_per_vector: The number of msecs to do an FFT over (at a time).
    :returns: Yields one FFT value and label at a time.
    """
    segment = segment.filter_silence()
    path, filename = os.path.split(path_to_segment)
    label = ClassLabels.NO_VOICE if "_NO" in path else ClassLabels.VOICE
    for frame, _timestamp in segment.generate_frames_as_segments(ms_per_vector):
        _bins, fft_vals = frame.fft()
        fft_vals = np.abs(fft_vals) / len(fft_vals)
        fft_vals = (fft_vals - min(fft_vals)) / (max(fft_vals) + 1E-9)
        yield fft_vals, label

def _generate_vector_by_vector(data_dir, sampling_frequency_hz, sample_width,
                               channels, ms_per_vector, ignore=None, include=None):
    """
    Generates the vectors and labels, one tuple at a time from data_dir, excluding directories
    from the ignore list. If the include list is specified, only it will be used.
    """
    generator = _generate_segments(data_dir,
                                   shuffle=True,
                                   sampling_frequency_hz=sampling_frequency_hz,
                                   sample_width=sample_width,
                                   channels=channels,
                                   ignore=ignore,
                                   include=include)

    # Collect five audio files's worth of data with labels
    # Shuffle them and release one tuple at a time
    # The reason for doing this is so that we can hopefully get more variance
    # into each batch than would otherwise be there
    for chunk in utils.grouper(5, generator):
        # chunk is a tuple of five (seg, path) tuples
        tups = [[tup for tup in _get_tups(segment, seg_path, ms_per_vector)] for segment, seg_path in chunk]
        tups_remain = sum([len(ls) for ls in tups]) > 0
        while tups_remain:
            tup_lists_with_remaining_items = [ls for ls in tups if len(ls) > 0]
            tup_list = random.choice(tup_lists_with_remaining_items)
            yield tup_list.pop(0)
            tups_remain = sum([len(ls) for ls in tups]) > 0

def generate_vectors_and_labels_from_file(fpath, msecs=30):
    """
    Takes a file path and generates vectors and labels from it until it is out of audio.

    The labels it generates are based purely on the labeling scheme of the directory structure.

    :param fpath: The path to the file.
    :param msecs: The number of msecs of audio to yield in the frequency domain at a time.
    """
    segment = audiosegment.from_file(fpath).resample(sample_rate_Hz=32000, channels=1, sample_width=2)
    for fft_val, label in _get_tups(segment, fpath, msecs):
        yield fft_val, label
