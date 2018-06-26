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

    batchsize = 16
    broke = False
    ms = 45
    total_segs_to_yield = 5 * 60 * 1000 / ms
    total_batches_to_yield = int(total_segs_to_yield / batchsize)
    batches = []
    for i, b in enumerate(provider.generate_n_fft_batches(n=None, batchsize=batchsize, ms=ms, label_fn=_label_fn, forever=True)):
        print("Batch", i, "out of", total_batches_to_yield)
        if i >= total_batches_to_yield:
            broke = True
            break
        else:
            batches.append(b)
    assert broke
    assert len(batches) == total_batches_to_yield