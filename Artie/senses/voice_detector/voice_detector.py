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

class VoiceDetector:
    """
    Class used for detecting voice in an audio stream. First, train it (or load it from a file),
    then feed it raw audio data, or use the audiosegment.detect_event function to detect voice
    in an audio stream.

    :param sample_rate_hz:      The sampling rate that this model expects the data to be in.
    :param sample_width_bytes:  The byte-width the model expects the data to be.
    :param ms:                  The number of ms of audio data to feed into the model at a time.
    :param model_type:          The type of model to create. Allowable options are: "fft", for a
                                model that uses `ms` of data at a time, transformed into the
                                frequency domain, and `spec` for a convolutional model that
                                transforms the data into a spectrogram before applying the model to it.
    :param window_length_ms:    Only used for spectrogram models. This is the number of ms per window, though
                                there will be a 50% overlap between each window.
    :param overlap:             Only used for spectrogram models. The fraction of each window to overlap.
    :param spectrogram_shape:   Only used for spectrogram models. The shape of each input spectrogram.
    """
    def __init__(self, sample_rate_hz=24_000, sample_width_bytes=2, ms=300, model_type="fft", window_length_ms=30, overlap=1/8, spectrogram_shape=None):
        self.sample_rate_hz = sample_rate_hz
        self.sample_width_bytes = sample_width_bytes
        self.ms = ms
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

    def fit(self, datagen, batch_size, **kwargs):
        """
        """
        tb = TensorBoard(log_dir='./logs', batch_size=batch_size, write_graph=True, write_grads=True)
        self._model.fit_generator(datagen, callbacks=[tb], **kwargs)
