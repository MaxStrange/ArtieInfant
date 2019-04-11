"""
The non-variational counterpart to vae.py in this package.
"""
from keras.models import Model
from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, MaxPooling2D, Flatten, Reshape

import keras
import numpy as np
import os

# TODO: This and the vae.py module share a lot of code. The two classes should have a common parent.

class AutoEncoder:
    """
    The non-variational analog to `VariationalAutoEncoder`. Should
    be mostly API compatible however, though the means and variances
    that we return will just be None.
    """
    def __init__(self, input_shape, latent_dim, optimizer, loss, encoder, decoder, inputlayer, decoderinputlayer, save_intermediate_models=False, tbdir=None):
        """
        """
        self._latent_dim = latent_dim
        self._encoder, self._inputs = self._build_encoder(latent_dim, encoder, inputlayer)
        self._decoder = self._build_decoder(decoder, decoderinputlayer)
        self._outputs = self._decoder(self._encoder(self._inputs))
        self._vae = Model(self._inputs, self._outputs, name='AutoEncoder')
        self._vae.compile(optimizer=optimizer, loss=loss)
        self._vae.summary()

        # Do callback initialization stuff
        if not os.path.isdir("models"):
            os.makedirs("models")
        if save_intermediate_models:
            saver = keras.callbacks.ModelCheckpoint("models/vae-weights.ep{epoch:02d}-loss{loss:.4f}.h5", period=1)
            self._callbacks = [saver]
        else:
            self._callbacks = []

        if tbdir:
            tb = keras.callbacks.TensorBoard(log_dir=tbdir,
                                                histogram_freq=0,   # every this many epochs, compute weights hist
                                                batch_size=32,      # Batch size for determining histogram
                                                write_graph=True,   # Should we write the whole network graph?
                                                write_grads=True,   # Gradient histograms
                                                write_images=False, # Not sure
                                                embeddings_freq=0,  # Every this many epochs, pass in embeddings_data to visualize embeddings
                                                update_freq=100)    # Every this many samples, we update tensor board
            self._callbacks.append(tb)

    def predict(self, *args, **kwargs):
        """
        """
        return self._decoder.predict(*args, **kwargs)

    def encode_decode(self, x, **kwargs):
        if len(x.shape) == 2:
            # Likely was passed a numpy array (rows, cols) - but we want a batch of images: (batch, rows, cols, channels)
            s = x.shape
            x = np.reshape(x, (1, s[0], s[1], 1))
        elif len(x.shape) == 3:
            # Who knows what we were given? Reject it and tell the user the error of their ways
            raise ValueError("Need either a numpy array of shape (rows, cols) or a batch of images (batch, rows, cols, channels). But got something with shape: {}".format(x.shape))
        return self._vae.predict(x, **kwargs)

    def fit(self, *args, **kwargs):
        """
        """
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = self._callbacks
        else:
            kwargs['callbacks'].extend(self._callbacks)

        return self._vae.fit(*args, **kwargs)

    def fit_generator(self, datagen, batch_size, save_models=True, **kwargs):
        """
        """
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = self._callbacks
        else:
            kwargs['callbacks'].extend(self._callbacks)

        return self._vae.fit_generator(datagen, **kwargs)

    def load_weights(self, weightfpath):
        """
        """
        self._vae.load_weights(weightfpath)

    def save_weights(self, fpath):
        """
        """
        self._vae.save_weights(fpath)

    def _build_encoder(self, latent_dim, encoder, inputlayer):
        """
        """
        latentspace = Dense(latent_dim, name='embedding_layer')(encoder)
        complete_encoder = Model(inputlayer, latentspace, name='encoder')
        complete_encoder.summary()
        return complete_encoder, inputlayer

    def _build_decoder(self, decoder, decoderinputlayer):
        """
        """
        complete_decoder = Model(decoderinputlayer, decoder, name='decoder')
        complete_decoder.summary()
        return complete_decoder
