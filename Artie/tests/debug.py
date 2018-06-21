"""
Scratch script used for debugging.
"""
import audiosegment
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
    #provider = fp.FeatureProvider(root, sample_rate=24_000, nchannels=1, bytewidth=2)
    provider = dp.DataProvider(root, sample_rate=24_000, nchannels=1, bytewidth=2)

    n = 15
    ms = 3000
    segs = [s for s in provider.generate_n_segments(n=n, ms=ms, batchsize=2)]
    print("Ultimately got:", len(segs), "segments out of", n)

    ms = 30
    furelise = [s for s in segs if "furelise" in s.name]
    furelise = furelise[0].reduce(furelise[1:])

    giggling = [s for s in segs if "giggling" in s.name]
    giggling = giggling[0].reduce(giggling[1:])

    laughter = [s for s in segs if "laughter" in s.name]
    laughter = laughter[0].reduce(laughter[1:])

    print(furelise)
    print(giggling)
    print(laughter)

    print("====================")

    furelise_raw = audiosegment.from_file("test_data_directory/furelise.wav")
    giggling_raw = audiosegment.from_file("test_data_directory/babies/baby_giggling.wav")
    laughter_raw = audiosegment.from_file("test_data_directory/babies/baby_laughter.wav")

    print(furelise_raw)
    print(giggling_raw)
    print(laughter_raw)