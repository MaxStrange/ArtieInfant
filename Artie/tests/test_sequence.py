"""
Tests the Multi-Processed Sequence implementation of the FeatureProvider.
"""
import numpy as np
import os
import sys
import unittest
import warnings
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import senses.dataproviders.featureprovider as fp # pylint: disable=locally-disabled, import-error
import senses.dataproviders.sequence as seq # pylint: disable=locally-disabled, import-error
import senses.voice_detector.voice_detector as vd # pylint: disable=locally-disabled, import-error

def _mb_to_ms(mb, bytewidth, sample_rate_hz):
    """
    Convert MB of single channel WAV file to ms.
    """
    total_bytes   = mb * 1E6
    total_samples = total_bytes / bytewidth
    total_seconds = total_samples / sample_rate_hz
    total_ms      = total_seconds * 1E3
    return total_ms

def _label_fn(fpath):
    if "babies" in fpath:
        return 0
    else:
        return 1

class TestSequence(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.root = os.path.abspath("test_data_directory")
        self.sample_rate = 24_000
        self.nchannels = 1
        self.bytewidth = 2
        mb_of_testdata = 28
        self.ms = 45
        self.batchsize = 32
        self.ms_of_dataset = _mb_to_ms(mb_of_testdata, self.bytewidth, self.sample_rate)
        self.ms_per_batch = self.ms * self.batchsize
        self.nworkers = 6
        self.provider = fp.FeatureProvider(self.root, sample_rate=self.sample_rate, nchannels=self.nchannels, bytewidth=self.bytewidth)
        args = (None, self.batchsize, self.ms, _label_fn)
        kwargs = {
            "normalize": True,
            "forever": True,
        }
        self.sequence = seq.Sequence(self.ms_of_dataset, 
                                     self.ms_per_batch,
                                     self.nworkers,
                                     self.root,
                                     self.sample_rate,
                                     self.nchannels,
                                     self.bytewidth,
                                     "generate_n_fft_batches",
                                     *args,
                                     **kwargs)

    def test_make_sequence(self):
        """
        Test the constructor.
        """
        pass  # Taken care of by setUp

    def test_get_one_batch(self):
        """
        Test basic functionality - get a single batch out of the Sequence.
        """
        for _ in range(1):
            batch = next(self.sequence)
        nbins = 541
        data_batch, label_batch = batch
        self.assertEqual(data_batch.shape, (self.batchsize, nbins))

        labels_that_are_ones = np.where(label_batch == 1)[0]
        labels_that_are_zeros = np.where(label_batch == 0)[0]
        self.assertEqual(len(labels_that_are_ones) + len(labels_that_are_zeros), len(label_batch))

    def test_train_with_sequence(self):
        """
        Test training the network using the Sequence class.
        """
        detector = vd.VoiceDetector(self.sample_rate, self.bytewidth, self.ms, "fft")
        detector.fit(self.sequence,
                     self.batchsize,
                     steps_per_epoch=50,
                     epochs=5,
                     use_multiprocessing=False,
                     workers=1)


if __name__ == "__main__":
    unittest.main()