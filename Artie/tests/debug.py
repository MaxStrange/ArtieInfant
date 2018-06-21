"""
Scratch script used for debugging.
"""
import audiosegment
import numpy as np
import os
import sys
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import senses.dataproviders.dataprovider as dp
import senses.dataproviders.featureprovider as fp


def _label_fn(fpath):
    if "babies" in fpath:
        return 0
    else:
        return 1

if __name__ == "__main__":
    root = os.path.abspath("test_data_directory")
    #provider = dp.DataProvider(root, sample_rate=24_000, nchannels=1, bytewidth=2)
    provider = fp.FeatureProvider(root, sample_rate=24_000, nchannels=1, bytewidth=2)

    n = None
    batchsize = 16
    ms = 45
    min_ffts_expected = 3 * 60 * 1000 / ms  # (minutes * sec/min * ms/sec) / ms/fft
    max_ffts_expected = 5 * 60 * 1000 / ms
    batches = [b for b in provider.generate_n_fft_batches(n=n, batchsize=batchsize, ms=ms, label_fn=_label_fn)]
    print("Number of FFT batches generated:", len(batches))
    assert len(batches) > 0
    assert len(batches) >= min_ffts_expected // batchsize
    assert len(batches) <= max_ffts_expected // batchsize
