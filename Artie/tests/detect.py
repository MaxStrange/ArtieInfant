"""
Script used to test out training the detectors.
"""
import audiosegment
import numpy as np
import os
import sys
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import senses.dataproviders.dataprovider as dp
import senses.dataproviders.featureprovider as fp
import senses.voice_detector.voice_detector as vd


def _label_fn(fpath):
    if "NOT_" in fpath:
        return 0
    else:
        return 1

if __name__ == "__main__":
    root = "/mnt/data/thesis_audio/baby_detection/processed"
    sample_rate = 24_000
    nchannels = 1
    bytewidth = 2
    provider = fp.FeatureProvider(root, sample_rate=sample_rate, nchannels=nchannels, bytewidth=bytewidth)
    validater = fp.FeatureProvider("/mnt/data/thesis_audio/baby_detection/test", sample_rate=sample_rate, nchannels=nchannels, bytewidth=bytewidth)

    n = None
    ms = 30
    batchsize = 32
    datagen = provider.generate_n_fft_batches(n, batchsize, ms, _label_fn, normalize=True, forever=True)
    validgen = validater.generate_n_fft_batches(n, batchsize, ms, _label_fn, normalize=True, forever=True)
    detector = vd.VoiceDetector(sample_rate_hz=sample_rate, sample_width_bytes=bytewidth, ms=ms, model_type="fft")
    detector.fit(datagen, batchsize, steps_per_epoch=100, epochs=200, validation_data=validgen, validation_steps=100, use_multiprocessing=False, workers=1)

    #n = None
    #ms = 300
    #batchsize = 32
    #shape = [s for s in provider.generate_n_spectrograms(n=1, ms=ms, label_fn=_label_fn, expand_dims=True)][0][0].shape
    #datagen = provider.generate_n_spectrogram_batches(n, batchsize, ms, _label_fn, normalize=True, forever=True, expand_dims=True)
    #validgen = validater.generate_n_spectrogram_batches(n, batchsize, ms, _label_fn, normalize=True, forever=True, expand_dims=True)
    #detector = vd.VoiceDetector(sample_rate_hz=sample_rate, sample_width_bytes=bytewidth, ms=ms, model_type="spec", window_length_ms=0.5, spectrogram_shape=shape)
    #detector.fit(datagen, batchsize, steps_per_epoch=100, epochs=200, validation_data=validgen, validation_steps=100, use_multiprocessing=True, workers=8)
