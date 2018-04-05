"""
TODO: Give this script a list of playlists and have it make a directory of raw
and test data. Don't do any preprocessing on the data - you can
have the octopod system do that.

When run as a script, this will mount the given hard drive at data/raw/
and then download all the data into it and create a test split.
"""
import audiosegment
import os
import shutil
import subprocess
import sys
from tqdm import tqdm
import traceback

def _make_test_split(target_path, name):
    """
    Makes a test split out of the data in the given path.
    Creates a test split by making a directory in target_path/test
    and then moving ~10% of the files it finds in target_path/processed
    into there.
    """
    # Get the file paths from target_path/name/processed
    processed_dir_path = target_path + "/" + name + "/processed"
    processed_fnames = [f for f in os.listdir(processed_dir_path) if os.path.isfile(processed_dir_path + "/" + f)]
    processed_fpaths = [processed_dir_path + "/" + fname for fname in processed_fnames]

    # Make this playlist's test dir
    test_dir_path = target_path + "/" + name + "/test"
    os.makedirs(test_dir_path, exist_ok=True)

    # Collect every tenth file in the directory
    test_split = processed_fpaths[::10]

    # Move the files into the test directory
    for fpath in test_split:
        fname = os.path.basename(fpath)
        test_path = test_dir_path + "/" + fname
        os.rename(fpath, test_path)

def _process_segments(new_segments, ascii_file_path, processed_path):
    """
    Processes a list of AudioSegments by
    resampling them to 48kHz, mono, 16bit and
    then saving them to 'path/processed/<whatever>_seg0.wav' etc.

    :param ascii_file_path: The path of the file used to create the `new_segments` list.
    :param processed_path:  'path/processed'
    """
    for i, new in enumerate(new_segments):
        new = new.resample(sample_rate_Hz=48000, channels=1, sample_width=2)
        new_name, _ext = os.path.splitext(os.path.basename(ascii_file_path))
        new_name = new_name + "_seg" + str(i) + ".wav"
        new_path = processed_path + "/" + new_name
        new.export(new_path, format="wav")
        del new

def _process_single_file(fname, path):
    """
    Processes a single file by:
    1. Cutting it up into 10 minute segments
    2. Saving those segments to 'path/processed/'
    """
    raw_file_path = path + "/" + fname
    ascii_file_path = "".join([i if ord(i) < 128 else 'x' for i in raw_file_path.replace(' ', '_')])
    try:
        segment = audiosegment.from_file(raw_file_path)
        new_segments = segment.dice(seconds=10 * 60)
        del segment
        _process_segments(new_segments, ascii_file_path, path + "/processed")
        del new_segments
    except OSError:
        print("OS ERROR while working on", fname)
        tb = traceback.format_exc()
        print(tb)
        pass  # Probably not enough RAM to fit the whole thing into memory. Just skip it.
    except MemoryError:
        print("MEMORY ERROR while working on", fname)
        tb = traceback.format_exc()
        print(tb)
        pass
    #os.remove(raw_file_path)

def _process_downloaded_playlist(path):
    """
    Processes the raw data found in the directory
    specified by `path` by:
    1. Creating a path/processed directory
    2. Cutting up each file in `path` into 10 minute segments
    3. Saving those segments, with ASCII-only, no-space names in path/processed.
    """
    if not os.path.isdir(path):
        print("!!", path, " is not a directory.")
        return

    # Make the 'processed' directory
    processed_path = path + "/processed"
    os.makedirs(processed_path, exist_ok=True)

    print("Working on directory", path)
    for fname in tqdm(os.listdir(path)):
        if os.path.isfile(path + "/" + fname):
            _process_single_file(fname, path)

def _download(target_path, name, url):
    """
    Downloads the playlist `name` found at `url`.
    Returns the path to the directory that the playlist
    was downloaded to.

    Note that this function makes no guarantees
    at all that it actually managed to do this.
    """
    path = target_path + "/" + name
    os.makedirs(path, exist_ok=True)

#   # Download the playlist to that directory
#   print("  |-> Executing youtube-dl on the playlist...")
#   dl_command = "youtube-dl --extract-audio --audio-format wav --yes-playlist --ignore-errors --max-filesize 3G "\
#                + url + " -o " + path + "/%(title)s-%(id)s.%(ext)s"
#   subprocess.run(dl_command.split(' '))
#   # Don't check result, who knows what youtube-dl returns
    return path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python3", sys.argv[0], "<path to conf file> <target path>")
        exit(1)

    _, config_path, target_path = sys.argv
    target_path = os.path.abspath(target_path)
    with open(config_path) as configfile:
        lines = [line.strip() for line in configfile if not line.strip().startswith('#')]

    names_and_urls = [map(lambda x: x.strip(), line.split(',')) for line in lines if line.strip()]
    for name, url in names_and_urls:
        print("Working on playlist:", name)
        path = _download(target_path, name, url)
        print("  |-> Path to process:", path)
        _process_downloaded_playlist(path)

    print("::::::::::::::::::::::::::")
    print(":: Creating test splits ::")
    print("::::::::::::::::::::::::::")
    names_and_urls = [map(lambda x: x.strip(), line.split(',')) for line in lines if line.strip()]
    for name, _url in names_and_urls:
        print("Working on", name)
        _make_test_split(target_path, name)

