"""
This module contains all the tests for the VAE.
"""
from keras.datasets import mnist
from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, MaxPooling2D, Flatten, Reshape
import numpy as np
import os
import sys
import test_sequence
import unittest
import warnings

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, path)
import internals.vae.vae as vae # pylint: disable=locally-disabled, import-error
import senses.dataproviders.featureprovider as fp # pylint: disable=locally-disabled, import-error
import senses.dataproviders.sequence as seq # pylint: disable=locally-disabled, import-error


class TestVAE(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("ignore", ResourceWarning)
        self.root = os.path.abspath("test_data_directory")
        self.sample_rate = 24_000
        self.nchannels = 1
        self.bytewidth = 2
        mb_of_testdata = 28
        self.ms = 45
        self.batchsize = 32
        self.ms_of_dataset = test_sequence.mb_to_ms(mb_of_testdata, self.bytewidth, self.sample_rate)
        self.ms_per_batch = self.ms * self.batchsize
        self.nworkers = 6
        self.provider = fp.FeatureProvider(self.root, sample_rate=self.sample_rate, nchannels=self.nchannels, bytewidth=self.bytewidth)

        args = (None, self.batchsize, self.ms, None)
        kwargs = {
            "normalize": True,
            "forever": True,
            "window_length_ms": None,
            "overlap": 0.5,
            "expand_dims": True,
        }
        self.sequence = seq.Sequence(self.ms_of_dataset,
                                     self.ms_per_batch,
                                     self.nworkers,
                                     self.root,
                                     self.sample_rate,
                                     self.nchannels,
                                     self.bytewidth,
                                     "generate_n_spectrogram_batches",
                                     *args,
                                     **kwargs)

    def _build_mnist_vae(self, latent_dim=2, optimizer="adam", loss="mse"):
        """
        Builds a default MNIST VAE and returns it.
        """
        input_shape = (28, 28, 1)

        # Encoder model
        inputs = Input(shape=input_shape, name="encoder_inputs")
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        x = MaxPooling2D((2, 2), padding='same')(x)                         # (-1, 14, 14, 16)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)         # (-1, 14, 14, 8)
        x = MaxPooling2D((2, 2), padding='same')(x)                         # (-1, 7, 7, 8)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)         # (-1, 7, 7, 8)
        x = MaxPooling2D((2, 2), padding='same')(x)                         # (-1, 4, 4, 8)
        x = Flatten()(x)                                                    # (-1, 128)
        encoder = Dense(32, activation='relu')(x)                           # (-1, 32)

        # Decoder model
        intermediate_dim = (4, 4, 8)  # hard-coded to MNIST
        decoderinputs = Input(shape=(latent_dim,), name="decoder_inputs")
        x = Dense(32, activation='relu')(decoderinputs)                      # (-1, 32)
        x = Dense(np.product(intermediate_dim), activation='relu')(x)        # (-1, 128)
        x = Reshape(target_shape=intermediate_dim)(x)                        # (-1, 4, 4, 8)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (-1, 4, 4, 8)
        x = UpSampling2D((2, 2))(x)                                          # (-1, 8, 8, 8)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (-1, 8, 8, 8)
        x = UpSampling2D((2, 2))(x)                                          # (-1, 16, 16, 8)
        x = Conv2D(16, (3, 3), activation='relu')(x)                         # (-1, 14, 14, 16)
        x = UpSampling2D((2, 2))(x)                                          # (-1, 28, 28, 16)
        decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # (-1, 28, 28, 1)

        return vae.VariationalAutoEncoder(input_shape, latent_dim, optimizer, loss, encoder=encoder, inputlayer=inputs, decoder=decoder, decoderinputlayer=decoderinputs)

    def _build_spectrogram_vae(self, latent_dim=10, optimizer="adam", loss="mse"):
        """
        Builds a default spectrogram VAE and returns it. The input dimensionality of this VAE is
        based on the Sequence created as part of setUp.
        """
        input_shape = (55, 19, 1)

        # Encoder model
        inputs = Input(shape=input_shape, name="encoder_inputs")
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)           # (-1, 55, 19, 16)
        x = MaxPooling2D((2, 2), padding='same')(x)                                 # (-1, 28, 10, 16)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)                 # (-1, 28, 10, 8)
        x = MaxPooling2D((2, 2), padding='same')(x)                                 # (-1, 14, 5, 8)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)                 # (-1, 14, 5, 8)
        x = MaxPooling2D((2, 2), padding='same')(x)                                 # (-1, 7, 3, 8)
        x = Flatten()(x)                                                            # (-1, 168)
        encoder = Dense(32, activation='relu')(x)                                   # (-1, 32)

        # Decoder model
        intermediate_dim = (14, 5, 8)
        decoderinputs = Input(shape=(latent_dim,), name="decoder_inputs")
        x = Dense(np.product(intermediate_dim), activation='relu')(decoderinputs)   # (-1, 560)
        x = Reshape(target_shape=intermediate_dim)(x)                               # (-1, 14, 5, 8)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)                 # (-1, 14, 5, 8)
        x = UpSampling2D((2, 2))(x)                                                 # (-1, 28, 10, 8)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)                 # (-1, 28, 10, 8)
        x = Conv2D(16, (2, 2), activation='relu', padding='same')(x)                # (-1, 28, 10, 16)
        x = UpSampling2D((2, 2))(x)                                                 # (-1, 56, 20, 16)
        decoder = Conv2D(1, (2, 2), activation='sigmoid', padding='valid')(x)       # (-1, 55, 19, 1)

        return vae.VariationalAutoEncoder(input_shape, latent_dim, optimizer, loss, encoder=encoder, inputlayer=inputs, decoder=decoder, decoderinputlayer=decoderinputs)

    def _get_mnist_data(self):
        """
        Returns the x_train and x_test split of the MNIST dataset,
        reshaped into flat np arrays of float32, normalized into [0.0, 1.0].
        """
        # Get the MNIST data, along with some parameters about it
        (x_train, _y_train), (x_test, _y_test) = mnist.load_data()
        original_dim = (28, 28, 1)

        # Reshape into the appropriate shapes and types
        x_train = np.reshape(x_train, [-1, *original_dim])
        x_test = np.reshape(x_test, [-1, *original_dim])
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        return x_train, x_test

    def test_make_vae_with_defaults(self):
        """
        Test simply creating a VAE with default values.
        """
        input_shape = (28, 28, 1)  # Must be this when encoder and decoder are None
        latent_dim = 2
        optimizer = "adam"
        loss = "mse"
        # Simply try to make it - that's all. We fail if it crashes.
        _autoencoder = vae.VariationalAutoEncoder(input_shape, latent_dim, optimizer, loss)

        # Create another VAE with slightly different args
        latent_dim = 7
        optimizer = "adadelta"
        _autoencoder = vae.VariationalAutoEncoder(input_shape, latent_dim, optimizer, loss)

    def test_make_vae_with_custom_args(self):
        """
        Test simply creating a VAE, but in this test, we create
        the encoder and the decoder externally and pass them in.
        """
        # Some misc variables
        latent_dim = 2
        optimizer = "adam"
        loss = "mse"

        _ = self._build_mnist_vae(latent_dim=latent_dim, optimizer=optimizer, loss=loss)

    @unittest.skipIf("TRAVIS_CI" in os.environ, "This test can't be run on Travis. Too much memory required.")
    def test_train_vae_to_completion_on_mnist(self):
        """
        Test creating and then training a simple VAE for MNIST.
        This test should fail only if there are bugs that prevent
        compiling the model or bugs that prevent the model from training
        on the data provided. We don't worry about loss score.
        """
        x_train, x_test = self._get_mnist_data()

        # Create and train the VAE
        vae = self._build_mnist_vae(latent_dim=2)
        vae.fit(x_train, x_train, epochs=2, batch_size=128, validation_data=(x_test, x_test))

    @unittest.skipIf("TRAVIS_CI" in os.environ, "This test can't be run on Travis. Too much memory required.")
    def test_train_vae_save_then_train_some_more(self):
        """
        Test creating, training, saving, loading, then training the VAE to
        completion. Uses the MNIST dataset.

        This is just a basic smoke test - it merely checks to make sure the above
        actions do not fail.
        """
        testpath = "__test_vae_weights_delete_me__.h5"
        x_train, x_test = self._get_mnist_data()
        vae = self._build_mnist_vae(latent_dim=2)
        vae.fit(x_train, x_train, epochs=1, batch_size=128, validation_data=(x_test, x_test))
        vae.save_weights(testpath)
        vae.load_weights(testpath)
        os.remove(testpath)
        vae.fit(x_train, x_train, epochs=1, batch_size=128, validation_data=(x_test, x_test))

    @unittest.skipIf("TRAVIS_CI" in os.environ, "This test can't be run on Travis. Too much memory required.")
    def test_train_vae_on_custom_data(self):
        """
        Test training the VAE on some custom data. The data should be awfully
        similar to what we are going to be using in the experiment. We don't
        care about loss score here, we just want to make a VAE that can train
        against what we actually want to train it against.
        """
        vae = self._build_spectrogram_vae()
        vae.fit_generator(self.sequence,
                          self.batchsize,
                          epochs=1,
                          save_models=False,
                          steps_per_epoch=50,
                          use_multiprocessing=False,
                          workers=1)

    @unittest.skipIf("TRAVIS_CI" in os.environ, "This test can't be run on Travis. Too much memory required.")
    def test_train_save_load_sample(self):
        """
        Test training the VAE on some custom data, then saving it, loading it,
        and then sampling some random values from latent space.

        For an example of this (with visualization!) using MNIST, see the vae.py's __main__.
        """
        vae = self._build_spectrogram_vae()
        _history = vae.fit_generator(self.sequence,
                                   self.batchsize,
                                   epochs=2,
                                   save_models=False,
                                   steps_per_epoch=50,
                                   use_multiprocessing=False,
                                   workers=1)
        sample = vae.sample()
        decoded_sample = vae.predict(np.array([sample]))
        self.assertEqual(decoded_sample.shape, (1, 55, 19, 1))

if __name__ == "__main__":
    unittest.main()
