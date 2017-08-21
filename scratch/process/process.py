"""
Master script for preprocessing the data.

This script is run once over all data in the "todo" directory every time new data comes in (after sorting from "raw/").
"""
import functools
import minutetrim
import multiprocessing.pool as pool
import os
import pydub
from segment import Segment
import shortfilter
import silencefilter
import sys
import vad

NOOP = lambda *args: None
DEBUG = NOOP#print

def get_days(paths_with_day_names):
    """
    Generator for getting each unique day from the path names. Path names are like this:
    path/to/file/YYYYMMDD_whatever.WAV

    Yields the YYYYMMDD part of the path string one at a time.
    """
    for path in paths_with_day_names:
        path_to_file_yyyymmdd_blah, _wav = os.path.splitext(path)
        yyyymmdd_blah = os.path.basename(path_to_file_yyyymmdd_blah)
        yyyymmdd = yyyymmdd_blah.split('_')[0]
        yield yyyymmdd

def reduce_day(segments, day_name, max_duration_minutes=None):
    """
    Takes a list of Segment objects (each of which has a name that includes their date) and a date string YYYYMMDD and
    concatenates each Segment object in the list which have the given date.

    If max_duration_minutes is specified as a positive integer, reduce_day will bite off a chunk of segments that
    add up to that number of minutes, reduce those into a single segment, then move on to the next chunk, and return
    the resulting segments. Importantly, if any segment in segments are larger than this number of minutes, these will
    be returned whole - not broken into pieces.

    Returns:
        The reduced segments if one or more segments match the day. None if none of them match.
    """
    def get_reducers_and_reducees(to_reduce, mins):
        reducees = []
        accumulator = 0
        for seg in to_reduce:
            accumulator += seg.duration_seconds / 60
            reducees.append(seg)
            if accumulator >= mins:
                if len(reducees) >= 2:
                    yield reducees[0], reducees[1:]
                else:
                    yield reducees[0], []
                reducees = []
                accumulator = 0
        if reducees:
            # yield whatever's left
            if len(reducees) >= 2:
                yield reducees[0], reducees[1:]
            else:
                yield reducees[0], []

    print("        |-> Getting each segment from day", day_name)
    to_reduce = [s for s in segments if day_name in s.name]
    print("        |-> Reducing list of length", len(to_reduce))
    if len(to_reduce) > 1 and max_duration_minutes is None:
        reduction = to_reduce[0]
        reductions = [reduction.reduce(to_reduce[1:])]
    elif len(to_reduce) > 1 and max_duration_minutes is not None:
        reductions = [reducer.reduce(r) for reducer, r in get_reducers_and_reducees(to_reduce, max_duration_minutes)]
    elif len(to_reduce) == 1:
        reductions = [to_reduce[0]]
    else:
        reductions = None

    print("        |-> Done reducing for day", day_name)
    return reductions

def split_segs_into_one_gb(sound_segs):
    """
    Splits the segments into lists of size 1GB ish.
    """
    ONE_GB = 1E9  # One GB is about 1 billion bytes
    this_seg = []
    acc = 0
    ret_segs = []
    for s in sound_segs:
        acc += len(s) * 4  # 4 bytes per float32
        if acc >= ONE_GB:
            acc = 0
            ret_segs += this_seg
            this_seg = []
        else:
            this_seg.append(s)
    return ret_segs

def process_day(day_folder, month_folder, year_folder, year_base_path):
    print(year_folder, month_folder, day_folder, ":")
    day_path = os.sep.join([year_base_path, year_folder, month_folder, day_folder])

    paths = [os.sep.join([day_path, day]) for day in os.listdir(day_path)]
    days = set([day for day in get_days(paths)])

    for p in paths:
        if "language" in p:
            # This directory has already been done. Move on
            return

    print("Converting each file into a pydub segment object. Num files:", str(len(paths)) + "...")
    segments = [Segment(pydub.AudioSegment.from_wav(p), p) for p in paths]
    DEBUG(segments)

    print("Applying processing pipeline...")

    print("    |-> Trimming each file down to one minute. Num files:", str(len(segments)) + "...")
    minute_trimmed_paths = minutetrim.minutetrim(segments)
    DEBUG(minute_trimmed_paths)

    print("    |-> Removing files that are shorter than", short_filter_duration, "seconds. Num files before filtering: " + str(len(minute_trimmed_paths)) + "...")
    short_filtered = shortfilter.shortfilter(minute_trimmed_paths, duration_seconds=short_filter_duration)
    DEBUG(short_filtered)

    print("    |-> Reducing to one sound file per 10 minutes per day...")
    short_filtered = [r for day in days for r in reduce_day(short_filtered, day, max_duration_minutes=10)]

    print("    |-> Splitting each wav file into silences and sounds (this one will take a while). Num files:", str(len(short_filtered)) + "...")
    sounds = silencefilter.silencefilter(short_filtered)
    DEBUG(sounds)
    sound_segs = []
    for ss in sounds:
        sound_segs.extend([s for s in ss])
    print("    |-> Removing files that are shorter than", short_filter_duration, "seconds...")
    sound_segs = shortfilter.shortfilter(sound_segs, duration_seconds=short_filter_duration)
    DEBUG(sound_segs)

    print("    |-> Detecting voice activity in the sound files. Num files:", str(len(sound_segs)) + "...")
    sound_segs = vad.vad(sound_segs)
    print("        |-> Filtering out the unvoiced sections...")
    sound_segs = [s for s in sound_segs if "voiced" in s.name]
    DEBUG(sound_segs)

    print("    |-> Reducing to one sound file per day...")
    final_segments = [r for day in days for r in reduce_day(sound_segs, day)]

    print("    |-> Writing each file to disk...")
    def save_speech_for_the_day(final_segments):
        print("        |-> Attempting to write...")
        for i, speech_for_that_day in enumerate(final_segments):
            name, _ext = os.path.splitext(speech_for_that_day.name)
            name_number = "" if i == 0 else str(i)  # This is for if we have several segments
            speech_for_that_day.export(name + "_language_" + name_number + ".WAV", format="WAV")
    try:
        raise OSError #######NOTE DEBUGGING TODO#######
        save_speech_for_the_day(final_segments)
    except OSError:
        print("!!!!!!!!!!!!!!!! OSERROR !!!!!!!!!!!!!")
        print("        |-> Attempting to deal with lots of data by splitting up into multiple arrays...")
        sound_segs = split_segs_into_one_gb(sound_segs)
        print("sound_segs:", sound_segs)
        final_segments = []
        for chunk in sound_segs:
            reduced_day = reduce_day(chunk, day)
            print("reduced_day:", reduced_day)
            #final_segments.append([r for day in days for r in reduce_day(chunk, day)])
        print("            |-> Number of chunks to write:", len(final_segments))
        save_speech_for_the_day(final_segments)

    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3", sys.argv[0], os.sep.join(["path", "to", "todo_directory"]))
        exit(-1)

    # Parameters
    short_filter_duration = 5  # Filter out files that are shorter than this number of seconds
#    process_pool = pool.Pool(processes=1)

    year_base_path = sys.argv[1]
    for year_folder in os.listdir(year_base_path):
        if year_folder == "2017" or year_folder == "2018":
            for month_folder in os.listdir(os.sep.join([year_base_path, year_folder])):
                process_day_folder = functools.partial(process_day, month_folder=month_folder, year_folder=year_folder, year_base_path=year_base_path)
#                process_pool.map(process_day_folder, os.listdir(os.sep.join([year_base_path, year_folder, month_folder])))
                for folder in os.listdir(os.sep.join([year_base_path, year_folder, month_folder])):
                    process_day_folder(folder)
