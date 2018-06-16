"""
This module provides functions for getting data out of a directory of WAV files.
"""
import audiosegment
import math
import os
import random


class DataProvider:
    def __init__(self, root, sample_rate=None, nchannels=None, bytewidth=None):
        """
        Crawls the given directory, recursively looking for WAV files. Does this by
        file extension ("WAV", ignoring case).

        :param root:        The root directory to crawl starting at.
        :param sample_rate: Will resample all audio files to this sample rate (Hz) before use.
        :param nchannels:   Will resample all audio files to this number of channels before use.
        :param bytewidth:   Will resample all audio files to this number of bytes data width before use.
        """
        self.root = root
        self.sample_rate = sample_rate
        self.nchannels = nchannels
        self.bytewidth = bytewidth
        self.path_cache = set()
        for root, _dirnames, fnames in os.walk(self.root):
            root_path = os.path.abspath(root)
            for fname in fnames:
                fpath = os.path.join(root_path, fname)
                if os.path.splitext(fname)[-1].lower() == ".wav":
                    self.path_cache.add(fpath)
        self._iterator_cache = set()  # We use this to keep track of which files we have already seen
        self._current_batch = []      # This is the current batch of WAV segments; if we ask for larger filebatch than needed, we have leftover

    def reset(self):
        """
        Resets the iterator, allowing calls to this class's functions to access segments that it
        has already seen.
        """
        self._iterator_cache.clear()

    def get_n_wavs(self, n):
        """
        Returns n random AudioSegment objects made from the wav files in
        this object's cache. If n is greater than the number of wav files
        in the cache, we return as many as we have.

        :param n:   The number of AudioSegment objects to return. If n is None, will return everything in the dataset.
        """
        wavs = [w for w in self.generate_n_wavs(n)]
        return wavs

    def generate_n_wavs(self, n):
        """
        Yields n random AudioSegments, one at a time, made from the wav files in
        this object's cache. If n is greater than the number of wav files in
        the cache, we return only as many as we have.

        :param n:   The number of AudioSegment objects to yield at most. If n is None, will return everything in the dataset.
        """
        n_yielded = 0
        for fpath in self.path_cache:
            if fpath in self._iterator_cache:
                continue
            if n is not None and n_yielded >= n:
                break
            else:
                seg = audiosegment.from_file(fpath)
                self._iterator_cache.add(fpath)
                if self.sample_rate is not None or self.nchannels is not None or self.bytewidth is not None:
                    n_yielded += 1
                    yield seg.resample(sample_rate_Hz=self.sample_rate, sample_width=self.bytewidth, channels=self.nchannels)
                else:
                    n_yielded += 1
                    yield seg

    def get_n_segments(self, n, ms, batchsize=10):
        """
        Returns n random AudioSegment objects of length ms each. The segments are chosen from
        a batch of `batchsize` files. The batch is chosen at random from the cache. 

        If the AudioSegment happens to fall at the end of the WAV file and does not line
        up neatly, it will be zero padded to reach `ms` length.
        """
        segs = [s for s in self.generate_n_segments(n=n, ms=ms, batchsize=batchsize)]
        return segs

    def _do_generate_n_segments(self, n, ms, batchsize):
        # Get a random batch of wavs
        wavs = self.get_n_wavs(batchsize)

        # Convert the ms to seconds
        seconds = ms / 1000

        # Chop up all the wavs into segments of `ms` length and add them to the collection of segments to draw from
        segments = []
        for wav in wavs:
            this_wav_segments = wav.dice(seconds, zero_pad=True)
            segments.extend(this_wav_segments)
        self._current_batch.extend(segments)

        # Now go through and cut up any left-over segments from previous calls that are longer than `ms`
        for seg in self._current_batch:
            if not math.isclose(len(seg), ms) and len(seg) > ms:
                pieces = seg.dice(seconds, zero_pad=True)
                self._current_batch.extend(pieces)

        # self._current_batch should now be composed of segments that are at most as long as `ms`, though there
        # might still be some left-over ones that are shorter - we will filter those out as we iterate through

        # Shuffle the batch of segments before iterating through them
        random.shuffle(self._current_batch)

        # Iterate through all the data and release only those that are the right number of `ms`
        for i, seg in enumerate(self._current_batch):
            if i >= n:
                self._current_batch = self._current_batch[i:]
                break
            elif not math.isclose(len(seg), ms):
                # This segment must be a left-over from a previous call with a different number of ms, let's just skip it
                continue
            else:
                yield seg

    def generate_n_segments(self, n, ms, batchsize=10):
        """
        Same as `get_n_segments`, but as a generator, rather than returning a whole list.
        """
        if n is None:
            yielded_some_this_time = True
            while yielded_some_this_time:
                items = [item for item in self._do_generate_n_segments(1000, ms, batchsize)]
                for item in items:
                    yield item
                nyielded = len(items)
                if nyielded > 0:
                    yielded_some_this_time = False
        else:
            yielded_so_far = 0
            while yielded_so_far < n:
                items = [item for item in self._do_generate_n_segments(n, ms, batchsize)]
                for item in items:
                    yield item
                yielded_so_far += len(items)
