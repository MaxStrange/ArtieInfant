"""
This is Unit Test module for the dataprovider.py module.
"""
import audiosegment
import os
import sys
import unittest
import warnings
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import senses.dataproviders.dataprovider as dp # pylint: disable=locally-disabled, import-error

class TestDataProvider(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.root = os.path.abspath("test_data_directory")
        self.nchannels = 1
        self.bytewidth = 2
        self.sample_rate = 24_000
        self.provider = dp.DataProvider(self.root, sample_rate=self.sample_rate, nchannels=self.nchannels, bytewidth=self.bytewidth)

    def reset(self):
        self.provider = dp.DataProvider(self.root, sample_rate=self.sample_rate, nchannels=self.nchannels, bytewidth=self.bytewidth)

    def test_get_zero_wavs(self):
        """
        Test trying to get 0 wav files from the data directory. Should
        simply return an empty list.
        """
        wavs = self.provider.get_n_wavs(0)
        self.assertEqual(len(wavs), 0)

    def test_get_one_wav(self):
        """
        Test getting one whole WAV file from the data directory.
        """
        wav = self.provider.get_n_wavs(1)
        self.assertEqual(len(wav), 1)

    def test_get_n_wav(self):
        """
        Test getting more than one whole WAV file from the data directory.
        """
        wavs = self.provider.get_n_wavs(2)
        self.assertEqual(len(wavs), 2)

    def test_resample(self):
        """
        Test that the data provider resamples to the correct values.
        """
        wavs = self.provider.get_n_wavs(3)
        for wav in wavs:
            self.assertEqual(wav.channels, self.nchannels)
            self.assertEqual(wav.sample_width, self.bytewidth)
            self.assertEqual(wav.frame_rate, self.sample_rate)

    def test_generate_n_wav(self):
        """
        Test yielding more than one whole WAV file from the data directory.
        """
        wavs = [w for w in self.provider.generate_n_wavs(2)]
        self.assertEqual(len(wavs), 2)

    def test_get_zero_segments(self):
        """
        Test getting 0 segments from the data directory. Should
        simply return an empty list.
        """
        segs = self.provider.get_n_segments(n=0, ms=30)
        self.assertEqual(len(segs), 0)

    def test_get_one_segment(self):
        """
        Test getting one 30ms segment from the data directory.
        """
        segs = self.provider.get_n_segments(n=1, ms=30, batchsize=2)
        self.assertEqual(len(segs), 1)
        self.assertEqual(len(segs[0]), 30)

    def test_get_n_segments(self):
        """
        Test getting more than one 30ms segment from the data directory.
        """
        segs = self.provider.get_n_segments(n=2, ms=30, batchsize=2)
        self.assertEqual(len(segs), 2)
        self.assertEqual(len(segs[0]), len(segs[1]))
        self.assertEqual(len(segs[0]), 30)

    def test_generate_n_segments(self):
        """
        Test generating more than one 30ms segment from the data directory.
        """
        segs = [s for s in self.provider.generate_n_segments(n=2, ms=30, batchsize=2)]
        self.assertEqual(len(segs), 2)
        self.assertEqual(len(segs[0]), len(segs[1]))
        self.assertEqual(len(segs[0]), 30)

    def test_generate_more_than_all_data(self):
        """
        Test to make sure we properly handle asking for more data than there is.
        """
        wavs = [w for w in self.provider.generate_n_wavs(n=25)]
        self.assertEqual(len(wavs), 3)

        self.reset()

        segs = [s for s in self.provider.generate_n_segments(n=15, ms=60_000, batchsize=2)]
        self.assertEqual(len(segs), 5)

        furelise = [s for s in segs if "furelise" in s.name]
        furelise = furelise[0].reduce(furelise[1:])
        self.assertGreaterEqual(len(furelise), 210_000)
        self.assertLessEqual(len(furelise), 215_000)

        giggling = [s for s in segs if "giggling" in s.name]
        giggling = giggling[0].reduce(giggling[1:])
        self.assertGreaterEqual(len(giggling), 36_000)
        self.assertLessEqual(len(giggling), 40_000)

        laughter = [s for s in segs if "laughter" in s.name]
        laughter = laughter[0].reduce(laughter[1:])
        self.assertGreaterEqual(len(laughter), 18_000)
        self.assertLessEqual(len(laughter), 20_000)

    def test_reset(self):
        """
        Test whether we can successfully reset the generator.
        """
        segs1 = [s for s in self.provider.generate_n_wavs(n=None)]
        self.reset()
        segs2 = [s for s in self.provider.generate_n_wavs(n=None)]
        self.assertEqual(len(segs1), len(segs2))
        self.assertEqual(len(segs1), 3)

        self.reset()
        segs1 = [s for s in self.provider.generate_n_segments(n=None, ms=1000, batchsize=2)]
        self.reset()
        segs2 = [s for s in self.provider.generate_n_segments(n=None, ms=1000, batchsize=2)]
        self.assertEqual(len(segs1), len(segs2))
        self.assertGreater(len(segs1), 0)

    def test_all_data_is_accounted_for(self):
        """
        Test to make sure that when we get all the data, all the data is accounted for.
        """
        ms = 30
        segs = [s for s in self.provider.generate_n_segments(n=None, ms=ms, batchsize=2)]
        furelise = [s for s in segs if "furelise" in s.name]
        furelise = furelise[0].reduce(furelise[1:])
        self.assertGreaterEqual(len(furelise), 210_000)
        self.assertLessEqual(len(furelise), 215_000)

        giggling = [s for s in segs if "giggling" in s.name]
        giggling = giggling[0].reduce(giggling[1:])
        self.assertGreaterEqual(len(giggling), 36_000)
        self.assertLessEqual(len(giggling), 40_000)

        laughter = [s for s in segs if "laughter" in s.name]
        laughter = laughter[0].reduce(laughter[1:])
        self.assertGreaterEqual(len(laughter), 18_000)
        self.assertLessEqual(len(laughter), 20_000)

    def test_no_duplicates(self):
        """
        Test to make sure we never get duplicate data.
        """
        _segs1 = [s for s in self.provider.generate_n_wavs(n=None)]
        segs2 = [s for s in self.provider.generate_n_wavs(n=None)]
        self.assertEqual(len(segs2), 0)

    def test_resume_iteration(self):
        """
        Test to make sure iteration can pick up where it leaves off across calls
        to generate_n_wavs.
        """
        segs1 = [s for s in self.provider.generate_n_wavs(n=1)]
        self.assertEqual(len(segs1), 1)
        segs2 = [s for s in self.provider.generate_n_wavs(n=2)]
        self.assertEqual(len(segs2), 2)
        segs3 = [s for s in self.provider.generate_n_wavs(n=None)]
        self.assertEqual(len(segs3), 0)

        self.reset()
        segs4 = [s for s in self.provider.generate_n_wavs(n=None)]
        self.assertEqual(len(segs4), 3)

    def test_iterate_through_all_data(self):
        """
        Test to make sure n=None iterates through all the data.
        """
        wavs = [s for s in self.provider.generate_n_wavs(n=None)]
        self.assertEqual(len(wavs), 3)

        self.reset()
        wavs = self.provider.get_n_wavs(n=None)
        self.assertEqual(len(wavs), 3)

        self.reset()
        ms = 1000
        segs = [s for s in self.provider.generate_n_segments(n=None, ms=ms, batchsize=2)]
        ln_ms_wavs = sum([len(w) for w in wavs])
        ln_ms_segs = sum([len(s) for s in segs])
        self.assertEqual(ln_ms_wavs // ms, ln_ms_segs // ms)

        self.reset()
        segs = self.provider.get_n_segments(n=None, ms=1000)
        ln_ms_segs = sum([len(s) for s in segs])
        self.assertEqual(ln_ms_wavs // ms, ln_ms_segs // ms)


if __name__ == "__main__":
    unittest.main()