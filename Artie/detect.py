"""
Script used to train the models.
"""
import argparse
import audiosegment
import numpy as np
import os
import sys
import senses.dataproviders.dataprovider as dp
import senses.dataproviders.featureprovider as fp
import senses.dataproviders.sequence as seq
import senses.voice_detector.voice_detector as vd

# These are how many GB of data are present in each root, recursively
root_sizes_in_gb = {
    "/mnt/data/thesis_audio/baby_detection/processed": 61,
    "/mnt/data/thesis_audio/engchin/processed": 338,
    "/mnt/data/thesis_audio/voice_detection/processed": 1300,

    "/mnt/data/thesis_audio/baby_detection/test": 1.5,
    "/mnt/data/thesis_audio/engchin/test": 13,
    "/mnt/data/thesis_audio/voice_detection/test": 30,
}

# These are the ratio of NEG/POS or CHINESE/ENG, by bytes of data (not number of files)
class_imbalances = {
    "/mnt/data/thesis_audio/baby_detection/processed": 3.55667,
    "/mnt/data/thesis_audio/engchin/processed": 5.6394,
    "/mnt/data/thesis_audio/voice_detection/processed": 1.3565,

    "/mnt/data/thesis_audio/baby_detection/test": 2.8887,
    "/mnt/data/thesis_audio/engchin/test": 4.43341,
    "/mnt/data/thesis_audio/voice_detection/test": 0.63616,
}

# These are the roots of the datasets for the various models I want to train
roots = {
    "BABY_DETECTION": "/mnt/data/thesis_audio/baby_detection/processed",
    "LANGUAGE_DETERMINATION": "/mnt/data/thesis_audio/engchin/processed",
    "VOICE_DETECTION": "/mnt/data/thesis_audio/voice_detection/processed",
}

# These are the roots of the validation sets for the various models I want to train
validation_roots = {
    "BABY_DETECTION": "/mnt/data/thesis_audio/baby_detection/test",
    "LANGUAGE_DETERMINATION": "/mnt/data/thesis_audio/engchin/test",
    "VOICE_DETECTION": "/mnt/data/thesis_audio/voice_detection/test",
}

def _gigabytes_to_ms(gb, sample_rate_hz, bytes_per_sample):
    """
    Approximately convert GB to ms of WAV data.
    """
    total_bytes   = gb * 1E9
    total_samples = total_bytes / bytes_per_sample
    total_seconds = total_samples / sample_rate_hz
    total_ms      = total_seconds * 1E3
    return total_ms

def _label_fn_engchin(fpath):
    """
    Function for labeling the data for the language determination model.
    """
    if "EN_" in fpath:
        return 0
    else:
        return 1

def _label_fn_binary(fpath):
    """
    Function for labeling the data for the other models.
    """
    if "NOT_" in fpath:
        return 0
    else:
        return 1

def _train_fft():
    """
    Train the model using the FFT method.
    """

if __name__ == "__main__":
    # Argparse to get what model domain and model specifics, plus any hyperparameters to be overridden
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs=1, type=str, choices=("baby", "language", "voice"), help="Model to train.")
    parser.add_argument("-sr", "--sample_rate", nargs=1, type=int, default=24E3, help="Sample rate in Hz to resample the data to.")
    parser.add_argument("-nc", "--nchannels", nargs=1, type=int, default=1, help="Number of audio channels to resample to.")
    parser.add_argument("-bw", "--bytewidth", nargs=1, type=int, default=2, help="Number of bytes per sample to resample the audio to.")
    parser.add_argument("-ms", "--ms_per_model_input", nargs=1, type=float, default=100.0, help="Number of ms of audio to use as the input to the model.")
    parser.add_argument("-nw", "--nworkers", nargs=1, type=int, default="Number of worker processes to use in preparing data to feed.")
    parser.add_argument("-pf", "--provider_fun", nargs=1, type=str, choices=("fft", "spectrogram", "sequence"), default="fft", help="The type of model to train.")
    parser.add_argument("-bs", "--batchsize", nargs=1, type=int, default=32, help="Batchsize for the minibatches to feed into model during training.")
    parser.add_argument("-fs", "--file_batchsize", nargs=1, type=int, default=10, help="The number of files to use as a batch for the underlying dataproviders.")
    parser.add_argument("-nz", "--normalize", nargs=1, type=bool, default=True, help="Only valid when provider_fun is 'fft' or 'spectrogram'. Normalize each FFT or spectrogram.")
    parser.add_argument("-so", "--spec_overlap", nargs=1, type=float, default=0.2, help="Only valid when provider_fun is 'spectrogram'. The fraction of overlap of each spectrogram window.")
    parser.add_argument("-sw", "--ms_per_spec_window", nargs=1, type=float, default=300, help="Only valid when provider_fun is 'spectrogram'. Length of each spectrogram in the time dimension.")
    args = parser.parse_args()

    # Convert user-friendly command line arguments to forms used internally
    modes = {"baby": "BABY_DETECTION", "language": "LANGUAGE_DETERMINATION", "voice": "VOICE_DETECTION"}
    args.mode = modes[args.mode]
    if args.provider_fun == "fft":
        args.provider_fun = "generate_n_fft_batches"
    elif args.provider_fun == "spectrogram":
        args.provider_fun = "generate_n_spectrogram_batches"
    else:
        args.provider_fun = "generate_n_sequence_batches"

    # Look up the arguments
    root                = roots[args.mode]
    validation_root     = validation_roots[args.mode]
    sample_rate_hz      = args.sample_rate
    nchannels           = args.nchannels
    bytewidth           = args.bytewidth
    batchsize           = args.batchsize
    ms_per_model_input  = args.ms_per_model_input
    ms_of_dataset       = _gigabytes_to_ms(root_sizes_in_gb[root], sample_rate_hz, bytewidth)
    ms_per_batch        = ms_per_model_input * batchsize
    nworkers            = args.nworkers
    provider_fun        = args.provider_function
    label_function      = _label_fn_engchin if "engchin" in root else _label_fn_binary
    file_batchsize      = args.file_batchsize
    normalize           = args.normalize
    spec_overlap        = args.spec_overlap
    ms_per_spec_window  = args.ms_per_spec_window
    args = (None, batchsize, ms_per_model_input, label_function)

    if provider_fun == "generate_n_fft_batches":
        kwargs = {"file_batchsize": file_batchsize, "normalize": normalize, "forever": True}
        model_type = "fft"
    elif provider_fun == "generate_n_spectrogram_batches":
        kwargs = {"file_batchsize": file_batchsize, "normalize": normalize, "forever": True, "window_length_ms": ms_per_spec_window, "overlap": spec_overlap, "expand_dims": True}
        model_type = "spec"
    elif provider_fun == "generate_n_sequence_batches":
        kwargs = {"file_batchsize": file_batchsize, "forever": True}
        model_type = "seq"
    else:
        assert False, "The function {} is not one of the available choices for FeatureProviders.".format(provider_fun)

    # Construct the FeatureProvider for validation
    validator = fp.FeatureProvider(validation_root,
                                   sample_rate=sample_rate_hz,
                                   nchannels=nchannels,
                                   bytewidth=bytewidth
                                  )

    # Construct the Sequence for training
    sequence = seq.Sequence(ms_of_dataset,
                            ms_per_batch,
                            nworkers,
                            root,
                            sample_rate_hz,
                            nchannels,
                            bytewidth,
                            provider_fun,
                            *args,
                            **kwargs
                            )

    # Construct the model
    if model_type == "spec":
        spectrogram_shape = [s for s in validator.generate_n_spectrograms(n=1, ms=ms_per_model_input, label_fn=label_function, expand_dims=True)][0][0].shape
    else:
        spectrogram_shape = None
    detector = vd.VoiceDetector(sample_rate_hz=sample_rate_hz,
                                sample_width_bytes=bytewidth,
                                ms=ms_per_model_input,
                                model_type=model_type,
                                window_length_ms=ms_per_spec_window,
                                overlap=spec_overlap,
                                spectrogram_shape=spectrogram_shape
                                )

    # Train the model
    detector.fit(sequence,
                 batch_size,
                 steps_per_epoch=ms_of_dataset/ms_per_batch,
                 epochs=n_epochs,
                 verbose=1,
                 validation_data=validator,
                 validation_steps=steps_in_validator,
                 class_weights=class_weights,
                 max_queue_size=10,
                 workers=1
                 use_multiprocessing=False,
                 shuffle=True,
                 initial_epoch=0
                 )
