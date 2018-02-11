"""
This node is a producer node. It generates data from
a directory tree by walking the tree, optionaly shuffling
the sound files it encounters, reading those files into
memory, optionally removing all silence,
cutting the files up into uniform length audio
segments, optionaly shuffling again, then publishing
them to a given topic for consumption by other nodes.
Each segment thus yielded is an AudioSegment object
whose name is the path to it on the harddrive.

Try --help for usage.
"""
import argparse
import audiosegment as asg
import collections
import logging
import myargparse
import mykafka
import os
import queue
import random
import threading

def _cache_file_paths(args):
    """
    Walks the directory given in args.target_dir and produces a list of fpaths of
    the audio files that we find. The returned list will be in the correct order
    based on args' 'shuffle' arguments.

    :param args: The args from parser.parse_args()
    :returns:    The (correctly shuffled) list of absolute paths to files that we can convert.
    """
    target_dir = args.target_dir[0]

    fpaths = []
    for root, dirs, files in os.walk(target_dir):
        local_fpaths = [os.path.join(root, f) for f in files]
        if args.shuffle_subdir:
            random.shuffle(local_fpaths)
        fpaths.extend(local_fpaths)
    if args.shuffle_tree:
        random.shuffle(fpaths)

    # Remove all duplicates
    return list(collections.OrderedDict.fromkeys(fpaths))

def _produce_segments(mailbox, args):
    """
    Slices the segments according to 'slice_length' in `args`
    and, if `args.shuffle_segments`, also shuffles the sliced
    segment before sending. Gets the segments from the `mailbox` as
    they come in. This function is the target for a second process.

    Produces the segments to the topics listed in `args.producer_topics`.

    :param args: The args from parser.parse_args()
    :returns:    None. If `args.forever`, does not return.
    """
    logging.info("Producing segments...")
    while True:
        seg = mailbox.get()
        logging.info("Produce thread got segment to send")
        slices = seg.dice(args.slice_length_s)
        if args.shuffle_segments:
            random.shuffle(slices)
        logging.debug("Sending slices...")
        for sl in slices:
            mykafka.produce(args.producer_topics, key=None, item=sl)
        if not args.forever:
            break

def _sanitize_args(args):
    """
    Sanity checks the given args. Exits with an error message if
    any check fails.

    :param args: The args from parser.parse_args()
    :returns:    None
    """
    target_dir = args.target_dir[0]
    # Make sure the target directory is actually a directory
    if not os.path.isdir(target_dir):
        print("Could not find or is not a directory:", target_dir, "; maybe check the spelling and/or use absolute path.")
        exit(1)

    # Make sure there are files in the target directory (or any of its subfiles)
    dir_contains_files = False
    for root, dirs, files in os.walk(target_dir):
        if len(files) != 0:
            dir_contains_files = True
            break
    if not dir_contains_files:
        print("Could not locate any files in the given directory", target_dir)
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=str, nargs=1, help="The target top-level directory that contains the audio files")
    parser.add_argument("--forever", action='store_true', help="Produces the whole batch of AudioSegments in a repeating loop forever until the end of time")
    parser.add_argument("--loglevel", type=str, choices={"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}, help="Set the level of logging to the console")
    parser.add_argument("--producer_topics", type=str, nargs="*", required=True, help="List of names of producer topics")
    parser.add_argument("--producer_configs", type=str, nargs="*", required=True, help="List of property=value strings")
    parser.add_argument("--remove_silence", action='store_true', help="Removes silence from the input audio files")
    parser.add_argument("--shuffle_tree", action='store_true', help="Shuffles all sound files in the directory")
    parser.add_argument("--shuffle_subdir", action='store_true', help="Suffles all sound files within each leaf directory")
    parser.add_argument("--shuffle_segments", action='store_true', help="Shuffles each file after slicing it")
    parser.add_argument("--shuffle_seed", type=int, default=2652, help="The random seed to use for the shuffling. Give a negative number for no seed")
    parser.add_argument("--slice_length", type=float, default=10.0, help="Length of each yielded AudioSegment in seconds", dest="slice_length_s")
    args = parser.parse_args()

    # Before anything else, start the logger
    logging.basicConfig(level=getattr(logging, args.loglevel.upper()), format="%(asctime)s %(message)s", datefmt="%m/%d %I:%M:%S %p")
    kafkalogger = logging.getLogger('kafka').setLevel("INFO")

    # Now get the producer configuration and start it up
    producer_configs = myargparse._parse_dict(args.producer_configs)
    mykafka.init_producer(**producer_configs)

    # Set the random seed if user wants to
    if args.shuffle_seed >= 0:
        random.seed(args.shuffle_seed)
        logging.info("Random seed set to " + str(args.shuffle_seed))

    # Do some basic error checking in the arguments
    _sanitize_args(args)

    # Walk the directory the user gives and find all the audio files
    logging.info("Walking the given path and searching for audio files...")
    cached_fpaths = _cache_file_paths(args)
    logging.info("Done caching the audio files")

    # Spin up a producer thread
    mailbox = queue.Queue()
    proc = threading.Thread(target=_produce_segments, args=(mailbox, args))
    proc.start()

    # Take each file, turn it into an AudioSegment, optionally remove silence, then send it to the producer thread
    while True:
        # Convert each audio file to AudioSegment
        logging.info("Converting each audiofile into audiosegment...")
        for f in cached_fpaths:
            seg = asg.from_file(f)
            if args.remove_silence:
                logging.info("Removing silence...")
                seg = seg.filter_silence()
                logging.info("Done removing silence")
            mailbox.put(seg)
        logging.info("Done converting the audiofiles into audiosegments")

        if not args.forever:
            break

    proc.join()
