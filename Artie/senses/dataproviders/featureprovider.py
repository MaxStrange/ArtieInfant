"""
This module is responsible for yielding data in the form
that wil be used in a pipeline or ML model.
"""
import senses.dataproviders.dataprovider as dataprovider
import numpy as np

class FeatureProvider:
    """
    Provides data in the form of Numpy Arrays with labels.
    """
    def __init__(self, root, sample_rate=None, nchannels=None, bytewidth=None, worker_index=None):
        """
        Passes `root` down to a DataProvider and encapsulates it.

        :param root:         The root of the data directory.
        :param sample_rate:  Will resample all audio files to this sample rate (Hz) before use.
        :param nchannels:    Will resample all audio files to this number of channels before use.
        :param bytewidth:    Will resample all audio files to this number of bytes data width before use.
        :param worker_index: If integer, will only take every worker_indexth item from the underlying dataset.
        """
        self.root           = root
        self.sample_rate    = sample_rate
        self.nchannels      = nchannels
        self.bytewidth      = bytewidth
        self.worker_index   = worker_index
        self.dp             = None
        self._reset()

    def _reset(self):
        """
        Resets the internal DataProvider.
        """
        if self.dp is not None:
            del self.dp
            self.dp = None
        self.dp = dataprovider.DataProvider(self.root, sample_rate=self.sample_rate, nchannels=self.nchannels, bytewidth=self.bytewidth, worker_index=self.worker_index)

    def generate_n_sequences(self, n, ms, label_fn, file_batchsize=10, forever=False):
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
        :param forever:         If True, ignore n and yield forever.
        :yields:                n tuples of the form (audio_samples, label)
        """
        if n is not None and n <= 0:
            return

        for seg in self.dp.generate_n_segments(n=n, ms=ms, batchsize=file_batchsize, forever=forever):
            label = label_fn(seg.name)
            samples = seg.get_array_of_samples()
            yield samples, label

    def generate_n_ffts(self, n, ms, label_fn, file_batchsize=10, normalize=True, forever=False):
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
        :param forever:         If True, yields FFTs forever, ignoring n.
        :yields:                n tuples of the form (FFT, label)
        """
        if n is not None and n <= 0:
            return

        for seg in self.dp.generate_n_segments(n=n, ms=ms, batchsize=file_batchsize, forever=forever):
            label = label_fn(seg.name)
            _hist_bins, hist_vals = seg.fft()
            real_normed = np.abs(hist_vals) / len(hist_vals)
            if normalize:
                real_normed = (real_normed - min(real_normed)) / (max(real_normed) + 1E-9)
            yield real_normed, label

    def generate_n_spectrograms(self, n, ms, label_fn, file_batchsize=10, normalize=True, window_length_ms=None, overlap=0.5, forever=False, expand_dims=False):
        """
        Yields n tuples of the form (label, Spectrogram), where:

        - labels are created from label_fn, by applying it to the file path (must be a function
          that takes a file path and returns a numeric value, which is the label). If label_fn is None,
          we actually simply copy the spectrograms and present those as the labels. This is what you
          want if you are training an autoencoder.
        - Spectrograms are created from AudioSegments of length ms each. The segments are chosen from a
          batch of `file_batchsize` files. The batch is chosen at random from the data directory.
          If the AudioSegment happens to fall at the end of the WAV file and does not line
          up neatly, it will be zero padded to reach `ms` length before being FFT'd. The FFTs that
          make up the spectrogram are real and normed.

        :param n:               The number of labeled spectrograms to yield
        :param ms:              The length of the AudioSegments that will be transformed, in ms
        :param label_fn:        Function of the signature fn(fpath) -> numeric label, or None for training
                                an autoencoder.
        :param file_batchsize:  The number of files to batch before creating AudioSegments from them
                                at random.
        :param normalize:       Maps the histogram values to between 0.0 and 1.0.
        :param window_length_ms: The length of time to accumulate for each FFT. If None, we take 1/100 of the time.
        :param overlap:         The fraction to overlap each FFT.
        :param forever:         If True, ignore n and yield forever.
        :param expand_dims:     If True, spectrograms will be of shape (nfreqbins, ntimebins, 1)
        :yields:                n tuples of the form (spectrogram, label)
        """
        if n is not None and n <= 0:
            raise ValueError("n must either be None or positive, but is {}".format(n))
        if window_length_ms is None:
            window_length_ms = ms / 10

        durs = []
        amplitudes_shapes = []
        for seg in self.dp.generate_n_segments(n=n, ms=ms, batchsize=file_batchsize, forever=forever):
            durs.append(seg.duration_seconds)
            assert len(set(durs)) == 1, "Got a duration that was not the same as all the others: {}".format(durs)

            _hist_bins, _times, amplitudes = seg.spectrogram(window_length_s=window_length_ms/1000, overlap=overlap)
            amplitudes_shapes.append(amplitudes.shape)
            assert len(set(amplitudes_shapes)) == 1, "Got a spectrogram that is not the same shape as the others: {}".format(amplitudes_shapes)

            amplitudes_real_normed = np.abs(amplitudes) / len(amplitudes)
            assert amplitudes_real_normed.shape == amplitudes.shape, "Normalization changed the shape from {} to {}.".format(amplitudes.shape, amplitudes_real_normed.shape)

            if normalize:
                amplitudes_real_normed = np.apply_along_axis(lambda v: (v - min(v)) / (max(v) + 1E-9), 1, amplitudes_real_normed)
            if expand_dims:
                amplitudes_real_normed = np.expand_dims(amplitudes_real_normed, -1)
            if label_fn is None:
                label = np.copy(amplitudes_real_normed)
            else:
                label = label_fn(seg.name)

            yield amplitudes_real_normed, label

    def generate_n_sequence_batches(self, n, batchsize, ms, label_fn, file_batchsize=10, forever=False):
        """
        Yields up to n batches of numpy arrays of the form:
        (batchsize, nsamples) along with a numpy array of labels.

        :param n:               The number of batches to yield. Will yield fewer if we don't have enough
        :param batchsize:       The number of sequences in a batch. Batches are composed of random sequences taken
                                from a cache batch of size file_batchsize.
        :param ms:              The length of the AudioSegments that will be transformed.
        :param label_fn:        Function of the signature fn(fpath) -> numeric label
        :param file_batchsize:  The number of files to batch before creating AudioSegments from them at random.
        :param forever:         If True, ignore n and yield forever.
        :yields:                Up to n tuples of the form (batch, labels), where batch is shaped: (batchsize, nsamples_of_audio_data)
        """
        if n is not None and n <= 0:
            return
        if batchsize <= 0:
            return

        nbatches_so_far = 0
        while n is None or nbatches_so_far < n:
            raw_batch = [tup for tup in self.generate_n_sequences(n=batchsize, ms=ms, label_fn=label_fn, file_batchsize=file_batchsize)]
            if not forever and not raw_batch:
                return
            elif forever and not raw_batch:
                # We are done with the data, but we want to keep going forever
                self._reset()
                continue
            elif len(raw_batch) != batchsize:
                continue
            sequences = np.reshape(np.array([seq for seq, _label in raw_batch]), (batchsize, -1))
            labels = np.array([label for _seq, label in raw_batch])
            yield sequences, labels
            nbatches_so_far += 1

    def generate_n_fft_batches(self, n, batchsize, ms, label_fn, file_batchsize=10, normalize=True, forever=False):
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
        :param forever:         If True, ignores n and yields forever.
        :yields:                Up to n tuples of the form (batch, labels), where batch is shaped: (batchsize, num_bins)
        """
        if n is not None and n <= 0:
            return
        if batchsize <= 0:
            return

        nbatches_so_far = 0
        while n is None or nbatches_so_far < n:
            raw_batch = [tup for tup in self.generate_n_ffts(n=batchsize, ms=ms, label_fn=label_fn, file_batchsize=file_batchsize, normalize=normalize)]
            if not forever and not raw_batch:
                # We are done with all the data
                return
            elif forever and not raw_batch:
                # We are done with all the data, but we want to yield forever, so just reset
                self._reset()
                continue
            elif len(raw_batch) != batchsize:
                continue
            ffts = np.reshape(np.array([fft for fft, _label in raw_batch]), (batchsize, -1))
            labels = np.array([label for _fft, label in raw_batch])
            yield ffts, labels
            nbatches_so_far += 1

    def generate_n_spectrogram_batches(self, n, batchsize, ms, label_fn, file_batchsize=10, normalize=True, window_length_ms=None, overlap=0.5, forever=False, expand_dims=False):
        """
        Yields up to n batches of numpy arrays of the form:
        (batchsize, num_freq_bins, num_time_bins)

        Each spectrogram is log10'd and then multiplied by 10.

        :param n:               The number of labeled spectrograms to yield
        :param batchsize:       The number of spectrograms in a batch. Batches are composed of random spectrograms
                                taken from a cache batch of size file_batchsize.
        :param ms:              The length of the AudioSegments that will be transformed, in ms
        :param label_fn:        Function of the signature fn(fpath) -> numeric label. If this is None, the label
                                will be identical to the batch, which is what you want if you are training an autoencoder.
        :param file_batchsize:  The number of files to batch before creating AudioSegments from them
                                at random.
        :param normalize:       Maps the histogram values to between 0.0 and 1.0.
        :param window_length_ms: The length of time to accumulate for each FFT. If None, we take 1/100 of the time.
        :param overlap:         The fraction to overlap each FFT.
        :param forever:         If True, ignore n and yield forever.
        :param expand_dims:     If True, expand the dimension of the batch to be (batchsize, nfbins, ntbins, 1)
        :yields:                Up to n tuples of the form (batch, label), where each batch is shaped: (batchsize, nfbins, ntbins)
        """
        if n is not None and n <= 0:
            raise ValueError("n must be either None or greater than zero but is {}".format(n))
        if batchsize <= 0:
            raise ValueError("batchsize must be greater than zero but is {}".format(batchsize))

        nbatches_so_far = 0
        while n is None or nbatches_so_far < n:
            raw_batch = [tup for tup in self.generate_n_spectrograms(batchsize, ms, label_fn, file_batchsize=file_batchsize, normalize=normalize, window_length_ms=window_length_ms, overlap=overlap, expand_dims=expand_dims)]
            if not forever and not raw_batch:
                return
            elif forever and not raw_batch:
                # We are out of data, but we need to reset
                self._reset()
                continue
            elif len(raw_batch) != batchsize:
                continue
            nfreqbins = raw_batch[0][0].shape[0]
            ntimebins = raw_batch[0][0].shape[1]
            shape = (batchsize, nfreqbins, ntimebins, 1) if expand_dims else (batchsize, nfreqbins, ntimebins)
            specs = np.reshape(np.array([spec for spec, _label in raw_batch]), shape)
            specs = 10.0 * np.log10(specs + 1e-9)
            labels = np.array([label for _spec, label in raw_batch])
            yield specs, labels
            nbatches_so_far += 1
