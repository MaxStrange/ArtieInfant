"""
This script computes descriptive statistics over the whole dataset and
produces artifacts for visualizing it.
"""
import argparse
import audiosegment as asg
import datetime
import math
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
        self.minfilelengthseconds = math.inf# The minimum file length in seconds
        self.maxfilelengthseconds = 0.0     # The maximum file length in seconds
        self.stdevfilelengthseconds = 0.0   # The standard deviation of file length in seconds

        self.avgsilenceseconds = 0.0        # The average number of seconds of silence in a file
        self.avgvoiceseconds = 0.0          # The average number of seconds of voice in a file
        self.avgotherseconds = 0.0          # The average number of seconds of other sound in a file

        self.filelengthsseconds = []        # A list of all the file lengths in seconds
        self.datetimes = []                 # The datetimes for each file we see
        self.startdate = None               # The datetime of the first recording
        self.enddate = None                 # The datetime of the last recording
        self.avgfilesperday = 0.0           # The average number of recordings per day
        self.stdevfilesperday = 0.0         # The standard deviation of recordings per day

    def add(self, seg: asg.AudioSegment, date: datetime.datetime) -> None:
        """
        Adds a single AudioSegment object's statistics. Also increments the count
        of total segments we have seen.
        """
        self.nfiles += 1
        segsecs = len(seg) / 1000.0
        self.totalseconds += segsecs

        # Silence
        nonsilence = seg.filter_silence(duration=5.0, threshold_percentage=5.0)
        self.totalsilenceseconds += (len(seg) - len(nonsilence)) / 1000.0

        # Voice
        voices = [s for s in seg.detect_voice() if s[0] == 'v']
        if len(voices) >= 2:
            voice = voices[0].reduce(voices[1:])
        elif len(voices) == 1:
            voice = voices[0]
        else:
            voice = None

        if voice:
            self.totalvoiceseconds += len(voice) / 1000.0

        # Other
        silence_ms = len(seg) - len(nonsilence)
        otherseconds = (len(seg) - (len(voice) + silence_ms)) / 1000.0
        if otherseconds > 0.0:
            self.totalotherseconds == otherseconds

        # Running minimum and maximum
        if segsecs < self.minfilelengthseconds:
            self.minfilelengthseconds = segsecs

        if segsecs > self.maxfilelengthseconds:
            self.maxfilelengthseconds = segsecs

        # File lengths
        self.filelengthsseconds.append(segsecs)

    def compute(self) -> None:
        """
        Does final computations on the dataset.
        """
        dates_to_nfiles = {}  # The number of files recorded on a day, for each day
        self.avgfilesperday = sum([n for n in dates_to_nfiles.values()]) / len(dates_to_nfiles.keys())
        self.avgfilelengthseconds = self.totalseconds / self.nfiles
        self.avgotherseconds = self.totalotherseconds / self.nfiles
        self.avgsilenceseconds = self.totalsilenceseconds / self.nfiles
        self.avgvoiceseconds = self.totalvoiceseconds / self.nfiles

        self.stdevfilelengthseconds = None
        self.stdevfilesperday = None

        sorteddates = sorted(self.datetimes)
        if sorteddates:
            self.startdate = sorteddates[0]
            self.enddate = sorteddates[-1]
        else:
            self.startdate = None
            self.enddate = None

    def describe(self) -> None:
        """
        Describes the dataset based on the statistics we have computed.
        Plots and does whatever else. You know.
        """
        pass

    def save(self, fpath: str) -> None:
        """
        Saves all of our stats to `fpath`.
        """
        pass

def parse_date_from_fname(fname: str) -> datetime.datetime:
    """
    Returns the datetime for the given recording, based on its name.
    """
    # TODO: See the dataset sorting script you used when importing the data in the first place
    return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rootdir', type=str, help="Root of the golden data directory.")
    parser.add_argument('savefname', type=str, default='dataset_descriptive_stats.txt', help="Place to output the descriptive statistics.")
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
                date = parse_date_from_fname(fname)
                stats.add(seg, date)

    stats.compute()
    stats.describe()
    stats.save(args.savefname)
