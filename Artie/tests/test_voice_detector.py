"""
Tests for the Voice Detector model(s).
"""
import numpy as np
import os
import shutil
import sys
import unittest
import warnings
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import senses.dataproviders.featureprovider as fp # pylint: disable=locally-disabled, import-error
import senses.voice_detector.voice_detector as vd # pylint: disable=locally-disabled, import-error

class TestVoiceDetector(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        logpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if os.path.isdir(logpath):
            shutil.rmtree(logpath)
        self.root = os.path.abspath("test_data_directory")
        self.sample_rate = 24_000
        self.nchannels = 1
        self.bytewidth = 2
        self.provider = fp.FeatureProvider(self.root, sample_rate=self.sample_rate, nchannels=self.nchannels, bytewidth=self.bytewidth)

    def _label_fn(self, fpath):
        """
        Returns 0 if fpath contains 'babies', otherwise 1.
        """
        if "babies" in fpath:
            return 0
        else:
            return 1

    def test_instantiate_vd_fft_instance(self):
        """
        Test creating the voice detector as an FFT model.
        """
        # Create the detector
        ms = 30
        detector = vd.VoiceDetector(sample_rate_hz=self.sample_rate, sample_width_bytes=self.bytewidth, ms=ms, model_type="fft")

        # Check that it knows how many inputs it should have
        nsamples = self.sample_rate * (ms / 1000)
        bins = np.arange(0, int(round(nsamples/2)) + 1, 1.0) * (self.sample_rate / nsamples)
        input_shape = (None, len(bins))
        self.assertEqual(input_shape, detector.input_shape)

    def test_instantiate_vd_spectrogram_instance(self):
        """
        Test creating the voice detector as a spectrogram model.
        """
        # Create the detector
        ms = 300
        nwindows = 10
        window_length_ms = ms / nwindows
        overlap = 0.5
        nfreqbins = 409  # TODO: Why is this 409?
        ntimebins = nwindows * (1/overlap) - 1
        input_shape = (int(nfreqbins), int(ntimebins), int(1))

        detector = vd.VoiceDetector(sample_rate_hz=self.sample_rate, sample_width_bytes=self.bytewidth, ms=ms, model_type="spec", window_length_ms=window_length_ms, overlap=overlap, spectrogram_shape=input_shape)

        # Check that it knows its input shape
        self.assertEqual((None, *input_shape), detector.input_shape)

    def test_fit_ffts(self):
        """
        Test training on FFT data.
        """
        n = None
        ms = 30
        batchsize = 32
        datagen = self.provider.generate_n_fft_batches(n, batchsize, ms, self._label_fn, normalize=True, forever=True)
        detector = vd.VoiceDetector(sample_rate_hz=self.sample_rate, sample_width_bytes=self.bytewidth, ms=ms, model_type="fft")
        detector.fit(datagen, batchsize, steps_per_epoch=100, epochs=2)

    def test_fit_spectrograms(self):
        """
        Test training on spectrogram data.
        """
        n = None
        ms = 300
        batchsize = 32
        shape = [s for s in self.provider.generate_n_spectrograms(n=1, ms=ms, label_fn=self._label_fn, expand_dims=True)][0][0].shape
        datagen = self.provider.generate_n_spectrogram_batches(n, batchsize, ms, self._label_fn, normalize=True, forever=True, expand_dims=True)
        detector = vd.VoiceDetector(sample_rate_hz=self.sample_rate, sample_width_bytes=self.bytewidth, ms=ms, model_type="spec", window_length_ms=0.5, spectrogram_shape=shape)
        detector.fit(datagen, batchsize, steps_per_epoch=100, epochs=2)


if __name__ == "__main__":
    unittest.main()