"""
Scratch script used for debugging.
"""
import audiosegment
import os
import sys
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import featureprovider as fp


def _label_fn(fpath):
    if "babies" in fpath:
        return 0
    else:
        return 1

if __name__ == "__main__":
    root = os.path.abspath("test_data_directory")
    provider = fp.FeatureProvider(root, sample_rate=24_000, nchannels=1, bytewidth=2)

    batchsize = 1
    ms = 5000
    n = 1
    batches = [batch for batch in provider.generate_n_spectrogram_batches(n=n, batchsize=batchsize, ms=ms, label_fn=_label_fn, window_length_ms=34)]
    batch = batches[0]
    spec = batch[1]
    print(spec.shape)
