"""
This script computes descriptive statistics over the whole dataset and
produces artifacts for visualizing it.
"""
import argparse
import audiosegment as asg
import datetime
import math
import os
import statistics

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

        self.dates_to_nfiles = {}           # A dict of the form {datetime: nfiles on that date}

    def add(self, seg: asg.AudioSegment, date: datetime.datetime) -> None:
        """
        Adds a single AudioSegment object's statistics. Also increments the count
        of total segments we have seen.
        """
        self.nfiles += 1
        segsecs = len(seg) / 1000.0
        self.totalseconds += segsecs

        # Silence
        nonsilence = seg.filter_silence(duration_s=5.0, threshold_percentage=5.0)
        self.totalsilenceseconds += (len(seg) - len(nonsilence)) / 1000.0

        # Voice
        voices = [s for (v, s) in seg.detect_voice() if v == 'v']
        if len(voices) >= 2:
            voice = voices[0].reduce(voices[1:])
            voicesecs = len(voice) / 1000.0
        elif len(voices) == 1:
            voice = voices[0]
            voicesecs = len(voice) / 1000.0
        else:
            voice = None
            voicesecs = 0.0

        self.totalvoiceseconds += voicesecs

        # Other
        voice_ms = voicesecs * 1000.0
        silence_ms = len(seg) - len(nonsilence)
        otherseconds = (len(seg) - (voice_ms + silence_ms)) / 1000.0
        if otherseconds > 0.0:
            self.totalotherseconds == otherseconds

        # Running minimum and maximum
        if segsecs < self.minfilelengthseconds:
            self.minfilelengthseconds = segsecs

        if segsecs > self.maxfilelengthseconds:
            self.maxfilelengthseconds = segsecs

        # File lengths
        self.filelengthsseconds.append(segsecs)

        # Datetime
        if date in self.dates_to_nfiles:
            self.dates_to_nfiles[date] += 1
        else:
            self.dates_to_nfiles[date] = 0

    def _compute_stdev(self, ls: [float], mean: float) -> float:
        """
        Computes the standard deviation of the given list of floats, given
        its mean. This only really makes sense for Gaussians.
        """
        return statistics.pstdev(ls, mu=mean)

    def compute(self) -> None:
        """
        Does final computations on the dataset.
        """
        # Averages
        self.avgfilesperday = sum([n for n in self.dates_to_nfiles.values()]) / len(self.dates_to_nfiles.keys())
        self.avgfilelengthseconds = self.totalseconds / self.nfiles
        self.avgotherseconds = self.totalotherseconds / self.nfiles
        self.avgsilenceseconds = self.totalsilenceseconds / self.nfiles
        self.avgvoiceseconds = self.totalvoiceseconds / self.nfiles

        # Standard deviations
        self.stdevfilelengthseconds = self._compute_stdev(self.filelengthsseconds, self.avgfilelengthseconds)
        self.stdevfilesperday = self._compute_stdev([n for n in self.dates_to_nfiles.values()], self.avgfilesperday)

        # Dates
        sorteddates = sorted(self.datetimes)
        if sorteddates:
            self.startdate = sorteddates[0]
            self.enddate = sorteddates[-1]
        else:
            self.startdate = None
            self.enddate = None

    def __str__(self):
        def printdurations(msg, secs):
            return "{} {} seconds, {} hours, {} days\n".format(msg, secs, secs / 3600.0, secs / (3600.0 * 24))

        s = "================= DATASET ================\n"
        s += "N Recordings: {}\n".format(self.nfiles)

        s += printdurations("Total Duration:", self.totalseconds)
        s += printdurations("Total Silence:", self.totalseconds)
        s += printdurations("Total Voice:", self.totalvoiceseconds)

        s += printdurations("Average Length of Each Recording:", self.avgfilelengthseconds)
        s += printdurations("Minimum Recording Length:", self.minfilelengthseconds)
        s += printdurations("Maximum Recording Length:", self.maxfilelengthseconds)
        s += printdurations("Standard Deviation of Recording Length:", self.stdevfilelengthseconds)

        s += printdurations("Average Amount of Silence in a Recording:", self.avgsilenceseconds)
        s += printdurations("Average Amount of Voice in a Recording:", self.avgvoiceseconds)
        s += printdurations("Average Amount of Non-Silence, Non-Voice Sound in a Recording:", self.avgotherseconds)

        s += "First date: {}\n".format(self.startdate)
        s += "Last date: {}\n".format(self.enddate)

        s += "Average Number of Recordings Per Day: {}\n".format(self.avgfilesperday)
        s += "Standard Deviation of Recordings Per Day: {}\n".format(self.stdevfilesperday)
        s += "==========================================\n"
        return s

    def describe(self) -> None:
        """
        Describes the dataset based on the statistics we have computed.
        Plots and does whatever else. You know.
        """
        print(self)

    def save(self, fpath: str) -> None:
        """
        Saves all of our stats to `fpath`.
        """
        with open(fpath, 'w') as f:
            f.write(self)

def parse_date_from_fname(fname: str) -> datetime.datetime:
    """
    Returns the datetime for the given recording, based on its name.
    """
    year = int(fname[:4])
    month = int(fname[4:6])
    day = int(fname[6:8])
    return datetime.datetime(year, month, day)

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
                seg = asg.from_file(fpath).resample(channels=1)
                date = parse_date_from_fname(fname)
                stats.add(seg, date)

    stats.compute()
    stats.describe()
    stats.save(args.savefname)
