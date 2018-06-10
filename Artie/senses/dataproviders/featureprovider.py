"""
This module is responsible for yielding data in the form
that wil be used in a pipeline or ML model.
"""
import dataprovider
import numpy as np

class FeatureProvider:
    """
    Provides data in the form of Numpy Arrays with labels.
    """
    def __init__(self, root, sample_rate=None, nchannels=None, bytewidth=None):
        """
        Passes `root` down to a DataProvider and encapsulates it.

        :param root:        The root of the data directory.
        :param sample_rate: Will resample all audio files to this sample rate (Hz) before use.
        :param nchannels:   Will resample all audio files to this number of channels before use.
        :param bytewidth:   Will resample all audio files to this number of bytes data width before use.
        """
        self.dp = dataprovider.DataProvider(root, sample_rate=sample_rate, nchannels=nchannels, bytewidth=bytewidth)

    def generate_n_sequences(self, n, ms, label_fn, file_batchsize=10):
        """
        Yields n tuples of the form (label, sequence), where:

        - labels are created from label_fn, by applying it to the file path (must be a function
          that takes a file path and returns a numeric value, which is the label)
        - Sequences are created from AudioSegments of length ms each. The segments are chosen from
          a batch of `file_batchsize` files. The batch is chosen at random from the data directory.
          If the AudioSegment happens to fall at the end of the WAV file and does not line up neatly,
          it will be zero padded to reach `ms` length before being returned.

        :param n:               The number of labeled sequences to yield.
        :param ms:              The length of the AudioSegments that will give out their raw samples.
        :param label_fn:        Function of the signature fn(fpath) -> numeric label
        :param file_batchsize:  The number of files to batch before creating AudioSegments from them
                                at random.
        :yields:                n tuples of the form (label, audio_samples)
        """
        for seg in self.dp.generate_n_segments(n=n, ms=ms, batchsize=file_batchsize):
            label = label_fn(seg.name)
            samples = seg.get_array_of_samples()
            yield label, samples

    def generate_n_ffts(self, n, ms, label_fn, file_batchsize=10, normalize=True):
        """
        Yields n tuples of the form (label, FFT), where:

        - labels are created from label_fn, by applying it to the file path (must be a function
          that takes a file path and returns a numeric value, which is the label)
        - FFTs are created from AudioSegments of length ms each. The segments are chosen from a
          batch of `file_batchsize` files. The batch is chosen at random from the data directory.
          If the AudioSegment happens to fall at the end of the WAV file and does not line
          up neatly, it will be zero padded to reach `ms` length before being FFT'd. These FFTs are
          real and normed.

        :param n:               The number of labeled FFTs to yield
        :param ms:              The length of the AudioSegments that will be transformed, in ms
        :param label_fn:        Function of the signature fn(fpath) -> numeric label
        :param file_batchsize:  The number of files to batch before creating AudioSegments from them
                                at random.
        :param normalize:       Maps the histogram values to between 0.0 and 1.0.
        :yields:                n tuples of the form (label, FFT)
        """
        for seg in self.dp.generate_n_segments(n=n, ms=ms, batchsize=file_batchsize):
            label = label_fn(seg.name)
            _hist_bins, hist_vals = seg.fft()
            real_normed = np.abs(hist_vals) / len(hist_vals)
            if normalize:
                real_normed = (real_normed - min(real_normed)) / (max(real_normed) + 1E-9)
            yield label, real_normed

    def generate_n_spectrograms(self, n, ms, label_fn, file_batchsize=10, normalize=True, window_length_ms=None, overlap=0.5):
        """
        Yields n tuples of the form (label, Spectrogram), where:

        - labels are created from label_fn, by applying it to the file path (must be a function
          that takes a file path and returns a numeric value, which is the label)
        - Spectrograms are created from AudioSegments of length ms each. The segments are chosen from a
          batch of `file_batchsize` files. The batch is chosen at random from the data directory.
          If the AudioSegment happens to fall at the end of the WAV file and does not line
          up neatly, it will be zero padded to reach `ms` length before being FFT'd. The FFTs that 
          make up the spectrogram are real and normed.

        :param n:               The number of labeled spectrograms to yield
        :param ms:              The length of the AudioSegments that will be transformed, in ms
        :param label_fn:        Function of the signature fn(fpath) -> numeric label
        :param file_batchsize:  The number of files to batch before creating AudioSegments from them
                                at random.
        :param normalize:       Maps the histogram values to between 0.0 and 1.0.
        :param window_length_ms: The length of time to accumulate for each FFT. If None, we take 1/100 of the time.
        :param overlap:         The fraction to overlap each FFT.
        :yields:                n tuples of the form (label, Spectrogram)
        """
        if window_length_ms is None:
            window_length_ms = ms / 10
        for seg in self.dp.generate_n_segments(n=n, ms=ms, batchsize=file_batchsize):
            label = label_fn(seg.name)
            _hist_bins, _times, amplitudes = seg.spectrogram(window_length_s=window_length_ms/1000, overlap=overlap)
            amplitudes_real_normed = np.abs(amplitudes) / len(amplitudes)
            if normalize:
                amplitudes_real_normed = np.apply_along_axis(lambda v: (v - min(v)) / (max(v) + 1E-9), 1, amplitudes_real_normed)

            yield label, amplitudes_real_normed

    def generate_n_sequence_batches(self, n, batchsize, ms, label_fn, file_batchsize=10):
        """
        Yields up to n batches of numpy arrays of the form:
        (batchsize, nsamples) along with a numpy array of labels.

        :param n:               The number of batches to yield. Will yield fewer if we don't have enough
        :param batchsize:       The number of sequences in a batch. Batches are composed of random sequences taken
                                from a cache batch of size file_batchsize.
        :param ms:              The length of the AudioSegments that will be transformed.
        :param label_fn:        Function of the signature fn(fpath) -> numeric label
        :param file_batchsize:  The number of files to batch before creating AudioSegments from them at random.
        :yields:                Up to n tuples of the form (label, batch), where batch is shaped: (batchsize, nsamples_of_audio_data)
        """
        nbatches_so_far = 0
        while nbatches_so_far < n:
            raw_batch = [tup for tup in self.generate_n_sequences(n=batchsize, ms=ms, label_fn=label_fn, file_batchsize=file_batchsize)]
            sequences = np.reshape(np.array([seq for _label, seq in raw_batch]), (batchsize, -1))
            labels = np.array([label for label, _seq in raw_batch])
            yield labels, sequences
            nbatches_so_far += 1

    def generate_n_fft_batches(self, n, batchsize, ms, label_fn, file_batchsize=10, normalize=True):
        """
        Yields up to n batches of numpy arrays of the form:
        (batchsize, num_bins) along with a numpy array of labels.

        :param n:               The number of batches to yield. Will yield fewer if we don't have enough
        :param batchsize:       The number of FFTs in a batch. Batches are composed of random FFTs taken
                                from a cache batch of size file_batchsize.
        :param ms:              The length of the AudioSegments that will be transformed.
        :param label_fn:        Function of the signature fn(fpath) -> numeric label
        :param file_batchsize:  The number of files to batch before creating AudioSegments from them at random.
        :param normalize:       Maps the histograms to between 0.0 and 1.0.
        :yields:                Up to n tuples of the form (label, batch), where batch is shaped: (batchsize, num_bins)
        """
        nbatches_so_far = 0
        while nbatches_so_far < n:
            raw_batch = [tup for tup in self.generate_n_ffts(n=batchsize, ms=ms, label_fn=label_fn, file_batchsize=file_batchsize, normalize=normalize)]
            ffts = np.reshape(np.array([fft for _label, fft in raw_batch]), (batchsize, -1))
            labels = np.array([label for label, _fft in raw_batch])
            yield labels, ffts
            nbatches_so_far += 1

    def generate_n_spectrogram_batches(self, n, batchsize, ms, label_fn, file_batchsize=10, normalize=True, window_length_ms=None, overlap=0.5):
        """
        Yields up to n batches of numpy arrays of the form:
        (batchsize, num_freq_bins, num_time_bins)

        :param n:               The number of labeled spectrograms to yield
        :param batchsize:       The number of spectrograms in a batch. Batches are composed of random spectrograms
                                taken from a cache batch of size file_batchsize.
        :param ms:              The length of the AudioSegments that will be transformed, in ms
        :param label_fn:        Function of the signature fn(fpath) -> numeric label
        :param file_batchsize:  The number of files to batch before creating AudioSegments from them
                                at random.
        :param normalize:       Maps the histogram values to between 0.0 and 1.0.
        :param window_length_ms: The length of time to accumulate for each FFT. If None, we take 1/100 of the time.
        :param overlap:         The fraction to overlap each FFT.
        :yields:                Up to n tuples of the form (label, batch), where each batch is shaped: (batchsize, nfbins, ntbins)
        """
        if n <= 0:
            return
        if batchsize <= 0:
            return

        nbatches_so_far = 0
        while nbatches_so_far < n:
            raw_batch = [tup for tup in self.generate_n_spectrograms(batchsize, ms, label_fn, file_batchsize=file_batchsize, normalize=normalize, window_length_ms=window_length_ms, overlap=overlap)]
            if not raw_batch:
                return
            nfreqbins = raw_batch[0][1].shape[0]
            ntimebins = raw_batch[0][1].shape[1]
            specs = np.reshape(np.array([spec for _label, spec in raw_batch]), (batchsize, nfreqbins, ntimebins))
            labels = np.array([label for label, _spec in raw_batch])
            yield labels, specs
            nbatches_so_far += 1