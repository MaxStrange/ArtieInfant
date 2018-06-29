"""
This module provides functions for getting data out of a directory of WAV files.
"""
import audiosegment
import math
import os
import random

class GeneratorError(Exception):
    def __init__(self):
        pass

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
        self.path_cache = []
        for root, _dirnames, fnames in os.walk(self.root):
            root_path = os.path.abspath(root)
            for fname in fnames:
                fpath = os.path.join(root_path, fname)
                if os.path.splitext(fname)[-1].lower() == ".wav":
                    self.path_cache.append(fpath)
        random.shuffle(self.path_cache)
        self._iterator_cache = set()  # We use this to keep track of which files we have already seen
        self._current_batch = []      # This is the current batch of WAV segments; if we ask for larger filebatch than needed, we have leftover

    def get_n_wavs(self, n):
        """
        Returns n random AudioSegment objects made from the wav files in
        this object's cache. If n is greater than the number of wav files
        in the cache, we return as many as we have.

        :param n:   The number of AudioSegment objects to return. If n is None, will return everything in the dataset.
        """
        if n is not None and (n >= len(self.path_cache)):
            n = None
        elif n is not None and (len(self._iterator_cache) + n > len(self.path_cache)):
            n = None

        if len(self.path_cache) == len(self._iterator_cache):
            return []
        else:
            wavs = [w for w in self.generate_n_wavs(n)]
            return wavs

    def generate_n_wavs(self, n, forever=False):
        """
        Yields n random AudioSegments, one at a time, made from the wav files in
        this object's cache. If n is greater than the number of wav files in
        the cache, we return only as many as we have.

        :param n:       The number of AudioSegment objects to yield at most. If n is None, will return everything in the dataset.
        :param forever: If True, will ignore n and simply generate data forever.
        """
        if n is not None and (n >= len(self.path_cache)):
            n = None
        elif forever:
            n = None

        n_yielded = 0
        i = 0

        # While we haven't seen all the data
        while len(self._iterator_cache) != len(self.path_cache):
            # If we have already yielded all we wanted, just return
            if n is not None and n_yielded >= n:
                break

            # Get the next path
            fpath = self.path_cache[i]

            # If we have already seen this path, skip it
            if fpath in self._iterator_cache:
                i += 1
                i = i % len(self.path_cache)
                continue

            # Load an AudioSegment into memory from the path
            seg = audiosegment.from_file(fpath)

            # Add the path to the ones we've seen
            self._iterator_cache.add(fpath)

            # If we need to resample it, do so; either way, yield it
            if self.sample_rate is not None or self.nchannels is not None or self.bytewidth is not None:
                n_yielded += 1
                yield seg.resample(sample_rate_Hz=self.sample_rate, sample_width=self.bytewidth, channels=self.nchannels)
            else:
                n_yielded += 1
                yield seg

            # If we are looping forever and we've now seen all the data, clear the cache of seen data
            if forever and n_yielded >= len(self.path_cache):
                self._iterator_cache.clear()

            # Increment i and wrap it around if we are done with the data and looping forever
            i += 1
            i = i % len(self.path_cache)

    def get_n_segments(self, n, ms, batchsize=10):
        """
        Returns n random AudioSegment objects of length ms each. The segments are chosen from
        a batch of `batchsize` files. The batch is chosen at random from the cache. 

        If the AudioSegment happens to fall at the end of the WAV file and does not line
        up neatly, it will be zero padded to reach `ms` length.
        """
        segs = [s for s in self.generate_n_segments(n=n, ms=ms, batchsize=batchsize)]
        return segs

    def _load_next_batch(self, ms, batchsize):
        # This function may take relatively quite long, due to potentially resampling wav files
        # So only do it periodically.
        if len(self._current_batch) >= 100:
            return len(self._current_batch)

        # Get a random batch of wavs
        wavs = self.get_n_wavs(batchsize)
        if not wavs and not self._current_batch:
            return 0

        # Convert the ms to seconds
        seconds = ms / 1000

        # Chop up all the wavs into segments of `ms` length and add them to the collection of segments to draw from
        segments = []
        for wav in wavs:
            this_wav_segments = wav.dice(seconds, zero_pad=True)
            segments.extend(this_wav_segments)
        self._current_batch.extend(segments)

        # Now go through and cut up any left-over segments from previous calls that are longer than `ms`
        to_extend_by = []
        for seg in self._current_batch:
            if not math.isclose(len(seg), ms) and len(seg) > ms:
                pieces = seg.dice(seconds, zero_pad=True)
                to_extend_by.extend(pieces)
        self._current_batch.extend(to_extend_by)

        # self._current_batch should now be composed of segments that are at most as long as `ms`, though there
        # might still be some left-over ones that are shorter - we will filter those out as we iterate through

        # Shuffle the batch of segments before iterating through them
        random.shuffle(self._current_batch)

        return len(self._current_batch)

    def _do_generate_n_segments(self, n, ms, batchsize):
        so_far_yielded = 0
        while self._load_next_batch(ms, batchsize):
            # Iterate through all the data and release only those that are the right number of `ms`
            for i, seg in enumerate(self._current_batch):
                assert len(seg) <= ms, "Length of segment is {} but expected no greater than {}".format(len(seg), ms)
                if n is not None and so_far_yielded >= n:
                    self._current_batch = self._current_batch[i:]
                    # If we are out of wav files and out of segments, we need to let the caller know by raising an exception
                    raise GeneratorError
                elif not math.isclose(len(seg), ms):
                    # This segment must be a left-over from a previous call with a different number of ms, let's just skip it
                    continue
                else:
                    yield seg
                    so_far_yielded += 1
            if n is None or so_far_yielded <= n:
                # If we are yielding everything, we should just dump any leftover tiny pieces in the batch
                self._current_batch = []

        # If we are out of wav files and out of segments, we need to let the caller know by raising an exception
        raise GeneratorError

    def generate_n_segments(self, n, ms, batchsize=10, forever=False):
        """
        Same as `get_n_segments`, but as a generator, rather than returning a whole list.

        :param forever:     If True, ignores n and yields data forever.
        """
        if forever:
            n = None

        if n is None:
            should_keep_iterating = True
            while should_keep_iterating:
                try:
                    for item in self._do_generate_n_segments(None, ms, batchsize):
                        yield item
                except GeneratorError:
                    pass
                should_keep_iterating = forever

                # If we want to keep iterating forever, we need to clear all the caches
                if should_keep_iterating:
                    self._iterator_cache.clear()
        else:
            yielded_so_far = 0
            try:
                while yielded_so_far < n:
                    for item in self._do_generate_n_segments(n, ms, batchsize):
                        yield item
                        yielded_so_far += 1
                        if yielded_so_far >= n:
                            self._current_batch = self._current_batch[yielded_so_far:]
                            break
            except GeneratorError:
                # We are out of stuff to yield
                return
