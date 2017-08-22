"""
Filters out silence from the wav files.
"""
import functools
import multiprocessing.pool as pool
import numpy as np
import os
from pyAudioAnalysis3 import audioSegmentation as audioseg
import pydub
from segment import Segment
import subprocess
import sys
import tempfile

MS_PER_S = 1000

def _filter_silence(ws, plot=False):
    Fs = ws.frame_rate
    X = np.array(ws.get_array_of_samples())
    window_size = 0.020
    step_size = 0.040
    smooth_window = 0.00001
    weight = 0.9
    segments = audioseg.silenceRemoval(X, Fs, window_size, step_size, smooth_window, weight, plot=plot)
    stops_silence = [segment[0] * MS_PER_S for segment in segments]
    starts_silence = [segment[1] * MS_PER_S for segment in segments]
    starts_sounds = [0] + [stop for stop in stops_silence]
    stops_sounds = [start for start in starts_silence] + [ws.duration_seconds * MS_PER_S]
    sounds = [ws[start:stop] for start, stop in zip(starts_sounds, stops_sounds)]

    return sounds

def _sox_filter_silence(ws, **kwargs):
    """
    Removes the silence from a given wave segment using Sox.

    Note: There are no keyword arguments. This function maintains backwards compatibility with the _filter_silence function.
    """
    tmp = tempfile.NamedTemporaryFile()
    othertmp = tempfile.NamedTemporaryFile()
    ws.export(tmp, format="WAV")
    command = "sox " + tmp.name + " " + othertmp.name + " silence 1 0.8 0.1% reverse silence 1 0.8 0.1% reverse"
    with subprocess.Popen(command.split(' '), stdout=subprocess.PIPE) as proc:
        print(proc.stdout.read())
        proc.wait()
        assert proc.returncode == 0, "Sox did not work as intended."
    ws = Segment(pydub.AudioSegment.from_wav(othertmp.name), ws.name)
    tmp.close()
    othertmp.close()

def silencefilter(wav_segments, plot=False):
    """
    Returns:
        [[sound files from wav_segments[0]], [sound files from wav_segments[1]], etc.]
    """

    func = functools.partial(_sox_filter_silence, plot=plot)
    process_pool = pool.Pool(processes=None)
    return process_pool.map(func, wav_segments)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3", sys.argv[0], os.sep.join(["path", "to", "file.wav"]))
        exit(-1)
    elif not os.path.isfile(sys.argv[1]):
        print(str(sys.argv[1]), "is not a valid path to a file.")
        exit(-1)

    fpath = sys.argv[1]
    print("Loading into memory as a Segment...")
    segment = Segment(pydub.AudioSegment.from_wav(fpath), fpath)
    print("Removing silence...")
    result = silencefilter([segment], plot=True)

