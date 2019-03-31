"""
This script computes descriptive statistics over the whole dataset and
produces artifacts for visualizing it.
"""
import argparse
import audiosegment as asg
import datetime
import os

class Statistics:
    """
    Class for keeping track of all the audiosegments. Add each segment one at a time.
    Then run compute() and describe().
    """
    def __init__(self):
        self.nfiles = 0                     # The number of audio files in the dataset

        self.totalseconds = 0.0             # The total number of seconds of audio in the dataset
        self.totalsilenceseconds = 0.0      # The total number of seconds that we consider to be silence
        self.totalvoiceseconds = 0.0        # The total number of seconds that we consider to be voice
        self.totalotherseconds = 0.0        # The total number of seconds that we consider to be other (not voice, not silence)

        self.avgfilelengthseconds = 0.0     # The average file length in seconds
        self.minfilelengthseconds = 0.0     # The minimum file length in seconds
        self.maxfilelengthseconds = 0.0     # The maximum file length in seconds
        self.stdevfilelengthseconds = 0.0   # The standard deviation of file length in seconds

        self.avgsilenceseconds = 0.0        # The average number of seconds of silence in a file
        self.avgvoiceseconds = 0.0          # The average number of seconds of voice in a file
        self.avgotherseconds = 0.0          # The average number of seconds of other sound in a file

        self.startdate = None               # The datetime of the first recording
        self.enddate = None                 # The datetime of the last recording
        self.avgfilesperday = 0.0           # The average number of recordings per day
        self.stdevfilesperday = 0.0         # The standard deviation of recordings per day

    def add(self, seg: asg.AudioSegment) -> None:
        """
        Adds a single AudioSegment object's statistics. Also increments the count
        of total segments we have seen.
        """
        pass

    def compute(self) -> None:
        """
        Does final computations on the dataset.
        """
        pass

    def describe(self) -> None:
        """
        Describes the dataset based on the statistics we have computed.
        Plots and does whatever else. You know.
        """
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rootdir', type=str, help="Root of the golden data directory.")
    args = parser.parse_args()

    if not os.path.isdir(args.rootdir):
        print("{} is not a valid directory.".format(args.rootdir))
        exit(1)

    stats = Statistics()
    for root, _subdirs, fnames in os.walk(args.rootdir):
        for fname in fnames:
            if os.path.splitext(fname)[-1].lower() in (".wav", ".ogg"):
                fpath = os.path.join(root, fname)
                print("Loading in {}...".format(fpath))
                seg = asg.from_file(fpath)
                stats.add(seg)

    stats.compute()
    stats.describe()
