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

    root = "/mnt/data/thesis_audio/baby_detection/processed"
    validation_root = "/mnt/data/thesis_audio/baby_detection/test"
    sample_rate = 24_000
    nchannels = 1
    bytewidth = 2
    provider = fp.FeatureProvider(root, sample_rate=sample_rate, nchannels=nchannels, bytewidth=bytewidth)
    validater = fp.FeatureProvider(validation_root, sample_rate=sample_rate, nchannels=nchannels, bytewidth=bytewidth)

    n = None
    ms = 30
    batchsize = 32
    ms_per_batch = batchsize * ms
    ms_in_dataset = _gigabytes_to_ms(root_sizes_in_gb[root], sample_rate, bytewidth)
    steps_per_epoch = ms_in_dataset / ms_per_batch
    ms_in_validation_set = _gigabytes_to_ms(root_sizes_in_gb[validation_root], sample_rate, bytewidth)
    steps_per_validation_epoch = ms_in_validation_set / ms_per_batch

    datagen = provider.generate_n_fft_batches(n, batchsize, ms, _label_fn, normalize=True, forever=True, file_batchsize=10)
    validgen = validater.generate_n_fft_batches(n, batchsize, ms, _label_fn, normalize=True, forever=True, file_batchsize=10)
    detector = vd.VoiceDetector(sample_rate_hz=sample_rate, sample_width_bytes=bytewidth, ms=ms, model_type="fft")
    detector.fit(datagen,
                 batchsize,
                 steps_per_epoch=steps_per_epoch,
                 epochs=25,
                 validation_data=validgen,
                 validation_steps=steps_per_validation_epoch,
                 use_multiprocessing=False,
                 workers=1)
