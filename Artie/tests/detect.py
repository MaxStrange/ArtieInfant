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
import senses.dataproviders.sequence as seq
import senses.voice_detector.voice_detector as vd

def _gigabytes_to_ms(gb, sample_rate_hz, bytes_per_sample):
    """
    Approximately convert GB to ms of WAV data.
    """
    total_bytes   = gb * 1E9
    total_samples = total_bytes / bytes_per_sample
    total_seconds = total_samples / sample_rate_hz
    total_ms      = total_seconds * 1E3
    return total_ms

def _label_fn(fpath):
    if "NOT_" in fpath:
        return 0
    else:
        return 1

if __name__ == "__main__":
    root_sizes_in_gb = {
        "/mnt/data/thesis_audio/baby_detection/processed": 61,
        "/mnt/data/thesis_audio/engchin/processed": 338,
        "/mnt/data/thesis_audio/voice_detection/processed": 1300,

        "/mnt/data/thesis_audio/baby_detection/test": 1.5,
        "/mnt/data/thesis_audio/engchin/test": 13,
        "/mnt/data/thesis_audio/voice_detection/test": 30,
    }

    # Dataset stuff
    root = "/mnt/data/thesis_audio/baby_detection/processed"
    validation_root = "/mnt/data/thesis_audio/baby_detection/test"

    # WAV stuff
    sample_rate = 24_000
    nchannels = 1
    bytewidth = 2

    # Hyperparameters
    ms = 45
    batchsize = 32
    nworkers = 6

    # Epoch Size and Batch Size
    ms_of_dataset = _gigabytes_to_ms(root_sizes_in_gb[root], sample_rate, bytewidth)
    ms_per_batch = ms * batchsize
    steps_per_epoch = ms_of_dataset / ms_per_batch
    n_epochs = 25

    # Sequence creation
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

    # Validator creation
    # TODO

    # Detector creation and training
    detector = vd.VoiceDetector(sample_rate, bytewidth, ms, "fft")
    detector.fit(sequence,
                 batchsize,
                 steps_per_epoch=steps_per_epoch,
                 epochs=n_epochs,
                 use_multiprocessing=False,
                 workers=1)
