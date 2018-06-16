"""
Tests for the Feature Provider
"""
import numpy as np
import os
import sys
import unittest
import warnings
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import featureprovider as fp

class TestFeatureProvider(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        root = os.path.abspath("test_data_directory")
        self.sample_rate = 24_000
        self.nchannels = 1
        self.bytewidth = 2
        self.provider = fp.FeatureProvider(root, sample_rate=self.sample_rate, nchannels=self.nchannels, bytewidth=self.bytewidth)

    def _label_fn(self, fpath):
        """
        Returns 0 if fpath contains 'babies', otherwise 1.
        """
        if "babies" in fpath:
            return 0
        else:
            return 1

    def test_generate_one_fft(self):
        """
        Tests yielding a single FFT of 45ms.
        """
        ffts = [(label, f) for label, f in self.provider.generate_n_ffts(n=1, ms=45, label_fn=self._label_fn)]
        self.assertEqual(len(ffts), 1)
        _label, fft = ffts[0]
        self.assertEqual(fft.shape, (541,))

    def test_generate_more_than_one_ffts(self):
        """
        Tests yielding more than one FFT of 34ms each.
        """
        ffts = [(label, f) for label, f in self.provider.generate_n_ffts(n=2, ms=34, label_fn=self._label_fn)]
        self.assertEqual(len(ffts), 2)
        for _label, fft in ffts:
            self.assertEqual(fft.shape, (409,))

    def test_generate_fft_minibatch(self):
        """
        Tests yielding several minibatches of labeled FFT data.
        """
        batchsize = 16
        batches = [batch for batch in self.provider.generate_n_fft_batches(n=5, batchsize=batchsize, ms=45, label_fn=self._label_fn)]
        self.assertEqual(len(batches), 5)

        label_batch, data_batch = batches[0]
        nbins = 541
        self.assertEqual(data_batch.shape, (batchsize, nbins))

        labels_that_are_ones = np.where(label_batch == 1)[0]
        labels_that_are_zeros = np.where(label_batch == 0)[0]
        self.assertEqual(len(labels_that_are_ones) + len(labels_that_are_zeros), len(label_batch))

    def test_generate_sequence_minibatch(self):
        """
        Tests yielding several minibatches of time-domain data.
        """
        batchsize = 32
        ms = 34
        n = 5
        batches = [batch for batch in self.provider.generate_n_sequence_batches(n=n, batchsize=batchsize, ms=ms, label_fn=self._label_fn)]
        self.assertEqual(len(batches), n)

        label_batch, data_batch = batches[0]
        datapoints = int(self.sample_rate * self.nchannels * (ms / 1000))
        self.assertEqual(data_batch.shape, (batchsize, datapoints))

        labels_that_are_ones = np.where(label_batch == 1)[0]
        labels_that_are_zeros = np.where(label_batch == 0)[0]
        self.assertEqual(len(labels_that_are_ones) + len(labels_that_are_zeros), len(label_batch))

    def test_generate_spectrogram_minibatch(self):
        """
        Tests yielding several minibatches of spectrogram data.
        """
        batchsize = 12
        ms = 340
        n = 3
        scalefactor = 10
        window_length = ms / scalefactor 
        batches = [batch for batch in self.provider.generate_n_spectrogram_batches(n=n, batchsize=batchsize, ms=ms, label_fn=self._label_fn, window_length_ms=window_length)]
        self.assertEqual(len(batches), n)

        label_batch, data_batch = batches[0]
        ndatapoints = scalefactor * 2 - 1  # with a 50% overlap
        nbins = 409
        self.assertEqual(data_batch.shape, (batchsize, nbins, ndatapoints))

        labels_that_are_ones = np.where(label_batch == 1)[0]
        labels_that_are_zeros = np.where(label_batch == 0)[0]
        self.assertEqual(len(labels_that_are_ones) + len(labels_that_are_zeros), len(label_batch))


if __name__ == "__main__":
    unittest.main()
