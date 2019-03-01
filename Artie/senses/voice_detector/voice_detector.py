"""
This module provides a VoiceDetector class, which provides a way to train
it and use it.
"""
import numpy as np
import os
import sys

import keras
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

class DataStatsCallback(keras.callbacks.Callback):
    """
    Callback for recording the spread of the data labels in a batch or epoch.
    """
    # TODO: Figure out how to get a breakdown of the labels to check on the split for each batch.
    #       May have to do it in the FeatureProvider instead.
    def __init__(self, print_to_screen=True, logfile=None):
        self._print_to_screen = print_to_screen
        self._logfile = logfile

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

        #if self._print_to_screen:
        #    print("Negative Samples:", len(neg), percent_negative)
        #    print("Positive Samples:", len(pos), percent_positive)

class VoiceDetector:
    """
    Class used for detecting voice in an audio stream. First, train it (or load it from a file),
    then feed it raw audio data, or use the audiosegment.detect_event function to detect voice
    in an audio stream.

    :param sample_rate_hz:      The sampling rate that this model expects the data to be in.
    :param sample_width_bytes:  The byte-width the model expects the data to be.
    :param ms:                  The number of ms of audio data to feed into the model at a time.
    :param normalize:           Should the data be normalized before prediction? Boolean.
    :param model_type:          The type of model to create. Allowable options are: "fft", for a
                                model that uses `ms` of data at a time, transformed into the
                                frequency domain, and `spec` for a convolutional model that
                                transforms the data into a spectrogram before applying the model to it.
    :param window_length_ms:    Only used for spectrogram models. This is the number of ms per window, though
                                there will be a 50% overlap between each window.
    :param overlap:             Only used for spectrogram models. The fraction of each window to overlap.
    :param spectrogram_shape:   Only used for spectrogram models. The shape of each input spectrogram.
    """
    def __init__(self, sample_rate_hz=24000, sample_width_bytes=2, ms=300, normalize=True, model_type="fft", window_length_ms=30, overlap=1/8, spectrogram_shape=None):
        self.sample_rate_hz = sample_rate_hz
        self.sample_width_bytes = sample_width_bytes
        self.ms = ms
        self.normalize = normalize
        self.model_type = model_type
        if self.model_type.strip().lower() == "fft":
            self._model = self._build_model_fft()
        elif self.model_type.strip().lower() == "spec":
            if not spectrogram_shape:
                raise ValueError("Need a spectrogram shape for a spectrogram model")
            self._model = self._build_model_spectrogram(spectrogram_shape)
        else:
            raise ValueError("Allowable models are 'fft' and 'spec'")
        self.window_length_ms = window_length_ms
        self.overlap = overlap

    @property
    def input_shape(self):
        """
        The shape this model expects for inputs.
        """
        return self._model.input_shape

    def _build_model_fft(self):
        """
        Builds and returns the compiled Keras network; uses FFTs
        as input data.
        """
        nsamples = self.sample_rate_hz * (self.ms / 1000)
        bins = np.arange(0, int(round(nsamples/2)) + 1, 1.0) * (self.sample_rate_hz / nsamples)

        model = Sequential()
        model.add(Dense(256, input_dim=len(bins), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        adam = Adam(lr=1E-3)
        model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
        return model

    def _build_model_spectrogram(self, input_shape):
        """
        Builds and returns the compiled Keras network; uses spectrograms
        as input data.
        """
        input_shape = [int(dim) for dim in input_shape]
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation="elu", input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation="elu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(Conv2D(64, (3, 3), activation="elu"))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        adam = Adam(lr=1E-3)
        model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
        return model

    def fit(self, datagen, batch_size, save_models=True, **kwargs):
        """
        """
        if not os.path.isdir("models"):
            os.makedirs("models")
        datastats_cb = DataStatsCallback()
        saver = keras.callbacks.ModelCheckpoint("models/weights.{epoch:02d}-{val_acc:.4f}.hdf5", period=1)
        callbacks = [datastats_cb, saver] if save_models else [datastats_cb]
        self._model.fit_generator(datagen, callbacks=callbacks, **kwargs)

    def predict(self, seg):
        """
        Returns a 0 if the event is not detected and 1 if the event is detected in the given segment.
        """
        if self.model_type == "fft":
            return self._predict_as_fft_model(seg)
        elif self.model_type == "spec":
            return self._predict_as_spec_model(seg)
        else:
            assert False, "Model type {} is impossible.".format(self.model_type)

    def _predict_as_fft_model(self, seg):
        """
        Returns a 0 if the event is not detected and 1 if the event is detected in the given segment.
        """
        _hist_bins, hist_vals = seg.fft()
        real_normed = np.abs(hist_vals) / len(hist_vals)
        if self.normalize:
            real_normed = (real_normed - min(real_normed)) / (max(real_normed) + 1E-9)
        prediction = self._model.predict(np.array([real_normed]), batch_size=1)
        prediction_as_int = int(round(prediction[0][0]))
        return prediction_as_int

    def _predict_as_spec_model(self, seg):
        """
        Returns a 0 if the event is not detected and 1 if the event is detected in the given segment.
        """
        _hist_bins, _times, amplitudes = seg.spectrogram(window_length_s=self.window_length_ms/1000, overlap=self.overlap)
        amplitudes_real_normed = np.abs(amplitudes) / len(amplitudes)
        if self.normalize:
            amplitudes_real_normed = np.apply_along_axis(lambda v: (v - min(v)) / (max(v) + 1E-9), 1, amplitudes_real_normed)
        amplitudes_real_normed = np.expand_dims(amplitudes_real_normed, axis=-1)  # Add batch dimension
        prediction = self._model.predict(np.array([amplitudes_real_normed]))
        prediction_as_int = int(round(prediction[0][0]))
        return prediction_as_int
