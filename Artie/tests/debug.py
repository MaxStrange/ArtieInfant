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
import senses.voice_detector.voice_detector as vd
import senses.dataproviders.sequence as seq

def _label_fn(fpath):
    if "babies" in fpath:
        return 0
    else:
        return 1

def _mb_to_ms(mb, bytewidth, sample_rate_hz):
    """
    Convert MB of single channel WAV file to ms.
    """
    total_bytes   = mb * 1E6
    total_samples = total_bytes / bytewidth
    total_seconds = total_samples / sample_rate_hz
    total_ms      = total_seconds * 1E3
    return total_ms

if __name__ == "__main__":
    #root = os.path.abspath("test_data_directory")
    #provider = dp.DataProvider(root, sample_rate=24_000, nchannels=1, bytewidth=2)
    #sample_rate = 24_000
    #nchannels = 1
    #bytewidth = 2
    #provider = fp.FeatureProvider(root, sample_rate=sample_rate, nchannels=nchannels, bytewidth=bytewidth)

    #n = None
    #ms = 30
    #batchsize = 32
    #datagen = provider.generate_n_fft_batches(n, batchsize, ms, _label_fn, normalize=True, forever=True)
    #detector = vd.VoiceDetector(sample_rate_hz=sample_rate, sample_width_bytes=bytewidth, ms=ms, model_type="fft")
    #detector.fit(datagen, batchsize, steps_per_epoch=100, epochs=2)

    #n = None
    #ms = 300
    #batchsize = 32
    #shape = [s for s in provider.generate_n_spectrograms(n=1, ms=ms, label_fn=_label_fn, expand_dims=True)][0][0].shape
    #datagen = provider.generate_n_spectrogram_batches(n, batchsize, ms, _label_fn, normalize=True, forever=True, expand_dims=True)
    #detector = vd.VoiceDetector(sample_rate_hz=sample_rate, sample_width_bytes=bytewidth, ms=ms, model_type="spec", window_length_ms=0.5, spectrogram_shape=shape)
    #detector.fit(datagen, batchsize, steps_per_epoch=100, epochs=2, use_multiprocessing=False, workers=2)

    ############ Sequence Testing ##############
    root = os.path.abspath("test_data_directory")
    sample_rate = 24_000
    nchannels = 1
    bytewidth = 2
    mb_of_testdata = 28
    ms = 45
    batchsize = 32
    ms_of_dataset = _mb_to_ms(mb_of_testdata, bytewidth, sample_rate)
    ms_per_batch = ms * batchsize
    nworkers = 6
    args = (None, batchsize, ms, _label_fn)
    kwargs = {
        "normalize": True,
        "forever": True,
    }
    sequence = seq.Sequence(ms_of_dataset, 
                            ms_per_batch,
                            nworkers,
                            root,
                            sample_rate,
                            nchannels,
                            bytewidth,
                            "generate_n_fft_batches",
                            *args,
                            **kwargs)

    for _ in range(1):
        batch = next(sequence)
    nbins = 541
    data_batch, label_batch = batch
    assert data_batch.shape == (batchsize, nbins)

    labels_that_are_ones = np.where(label_batch == 1)[0]
    labels_that_are_zeros = np.where(label_batch == 0)[0]
    assert len(labels_that_are_ones) + len(labels_that_are_zeros) == len(label_batch)

