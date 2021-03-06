"""
This module contains all the code necessary for the VAE.

The general use is to instantiate a VAE with the appropriate
hyper parameters, then to train it with a dataset. You can
then load the resulting weights into a VAE later.

Much of this code was taken from here: https://blog.keras.io/building-autoencoders-in-keras.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, MaxPooling2D, Flatten, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import numpy as np
import argparse
import os

def _sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.

    This function combines `args` (which must be a mean and a log of variance)
    into a single distribution. We then sample from that distribution to
    produce a latent vector.

    :param args:    Two tensors of shape (batch, nembedding_dims) - the first of which
                    is the means and the second of which is the log of the variances.
    :returns:       z: The sampled latent vector.
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]

    epsilon = K.random_normal(shape=(batch, dim))
    stdev = K.exp(0.5 * z_log_var)

    #z_mean = K.print_tensor(z_mean, message="Z-Mean")
    #z_log_var = K.print_tensor(z_log_var, message="Z Log Var")
    #epsilon = K.print_tensor(epsilon, message="E")
    #stdev = K.print_tensor(stdev, message="STDEV")

    sample = z_mean + stdev * epsilon
    #sample = K.print_tensor(sample, message="Sample")

    return sample

class VariationalAutoEncoder:
    """
    A class to represent an encoder and decoder such that the encoder
    is constrained to learning an embedding of the input vectors
    that conforms to a normal distribution.
    """
    def __init__(self, input_shape, latent_dim, optimizer, loss, *,
                        kl_loss_prop=0.5, recon_loss_prop=0.5, std_loss_prop=0.0, encoder=None, decoder=None, inputlayer=None,
                        decoderinputlayer=None, save_intermediate_models=False, tbdir=None):
        """
        :param input_shape:                 (tuple) The shape of the input data.
        :param latent_dim:                  (int) The dimensionality of the latent vector space.
        :param optimizer:                   String representation of the optimizer.
        :param loss:                        String representation of the loss function.
        :param kl_loss_prop:                Value between 0 and 1.0 that shows how much of the whole VAE loss function to assign to KL loss
        :param recon_loss_prop:             Value between 0 and 1.0 that shows how much of the whole VAE loss function to
                                            assign to reconstructive loss
        :param std_loss_prop:               Value between 0 and 1.0 that shows how much of the whole VAE loss function
                                            to assign to the variance portion
        :param encoder:                     The encoder layer. This must be the result of a sequence of Keras functional calls.
                                            You will also need to pass in the encoder's inputlayer.
                                            If None provided, we use a reasonable default for the MNIST dataset.
                                            This is probably not what you want.
        :param decoder:                     The decoder layer. This must be the result of a sequence of Keras functional calls.
                                            You will also need to pass in decoderinputlayer. The input shape must be (latent_dim,)
                                            and the output shape must be `input_shape`.
                                            If None provided, we use a reasonable default for the MNIST dataset.
                                            This is probably not what you want.
        :param inputlayer:                  Necessary if encoder is not None. This must be a Keras Inputs layer.
        :param decoderinputlayer:           Necessary if decoder is not None. This must be a Keras Inputs layer.
        :param save_intermediate_models:    If `True`, we will save the most recent model after every epoch in a 'models' directory.
        :param tbdir:                       If truthy, must be a directory. This directory will be where we save the tensorboard information.
                                            If None or empty, we will not use tensorboard. To use tensorboard, simply run it from the command
                                            line and pass in --logdir=full/path/to/tbdir
        """
        if encoder is not None and inputlayer is None:
            raise ValueError("If `encoder` is not None, you must also pass in `inputlayer`.")
        if decoder is not None and decoderinputlayer is None:
            raise ValueError("If `decoder` is not None, you must also pass in `decoderinputlayer`.")

        self._latent_dim = latent_dim
        self._encoder, self._inputs, z_mean, z_log_var = self._build_encoder(input_shape, latent_dim, encoder=encoder, inputlayer=inputlayer)
        self._decoder = self._build_decoder(input_shape, latent_dim, decoder=decoder, decoderinputs=decoderinputlayer)
        self._outputs = self._decoder(self._encoder(self._inputs)[2])
        self._vae = Model(self._inputs, self._outputs, name='vae')
        flattened_input_shape = (np.product(np.array(input_shape)),)

        def _vae_loss(y_true, y_pred):
            """
            VAE loss that is broken out as a separate function. It is required to be broken
            out as a separate function for reasons not well understood. See:
            https://github.com/keras-team/keras/issues/10137
            """
            reconstruction_loss = self._build_loss(loss, flattened_input_shape)
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            stddev_loss = K.mean(K.square(z_log_var))
            vae_loss = K.mean((reconstruction_loss * recon_loss_prop) + (kl_loss * kl_loss_prop) + (stddev_loss * std_loss_prop))

            return vae_loss

        self._vae.compile(optimizer=optimizer, loss=_vae_loss)
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

    def sample(self):
        """
        Returns a vector sampled from the latent VAE space. If you want to decode it using predict(),
        you must wrap it in a list:

        ```python
        sample = vae.sample()
        decoded_sample = vae.predict([sample])
        ```
        """
        mu = 0.0
        sigma = 1.0
        size = self._latent_dim
        return np.random.normal(mu, sigma, size)

    def sample_from_gaussian(self, mu: tuple, sigma: tuple):
        """
        Same as `sample` but instead of sampling from the normal distribution, samples from the given
        Gaussian.

        mu and sigma must be tuples of floats, of shape self._latent_dim.
        """
        return np.random.normal(mu, sigma)

    def predict(self, *args, **kwargs):
        """
        Do inference with the VAE. See the Keras predict() method's documentation: https://keras.io/models/model/#predict
        This method runs through the decoder specifically. The encoder is unused.
        To run something through the entire model, use `encode_decode()`.
        """
        return self._decoder.predict(*args, **kwargs)

    def encode_decode(self, x, **kwargs):
        """
        Do inference with the VAE. See the Keras predict() method's documentation: https://keras.io/models/model/#predict
        This method runs through the whole encoder and decoder. To just use the decoder, use `predict()`.
        """
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
        Train the VAE on the given data. See the Keras fit() method's documentation: https://keras.io/models/model/#fit
        """
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = self._callbacks
        else:
            kwargs['callbacks'].extend(self._callbacks)

        return self._vae.fit(*args, **kwargs)

    def fit_generator(self, datagen, batch_size, save_models=True, **kwargs):
        """
        Train the VAE on the given Sequence (data generator).
        """
        if 'callbacks' not in kwargs:
            kwargs['callbacks'] = self._callbacks
        else:
            kwargs['callbacks'].extend(self._callbacks)

        return self._vae.fit_generator(datagen, **kwargs)

    def load_weights(self, weightfpath):
        """
        Loads the given fpath and loads the weights found in that file.
        """
        self._vae.load_weights(weightfpath)

    def save_weights(self, fpath):
        """
        Saves the weights of this model at the given file path.
        """
        self._vae.save_weights(fpath)

    def _build_loss(self, loss, flattened_input_shape):
        """
        Builds and returns the loss function.
        """
        newshape = (-1, *flattened_input_shape)
        if loss.lower().strip() == "mse":
            reconstruction_loss = mse(K.reshape(self._inputs, newshape), K.reshape(self._outputs, newshape))
        else:
            raise ValueError("{} is not an allowed loss function.".format(loss))
        reconstruction_loss *= flattened_input_shape
        return reconstruction_loss

    def _build_encoder(self, input_shape, latent_dim, encoder=None, inputlayer=None):
        """
        Builds the encoder portion of the model and returns it.

        The default encoder is good for MNIST.
        """
        #                                                                       # MNIST dimensionality
        if encoder is None:
            inputs = Input(shape=input_shape, name="encoder_input")             # (-1, 28, 28, 1)
            x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)   # (-1, 28, 28, 16)
            x = MaxPooling2D((2, 2), padding='same')(x)                         # (-1, 14, 14, 16)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)         # (-1, 14, 14, 8)
            x = MaxPooling2D((2, 2), padding='same')(x)                         # (-1, 7, 7, 8)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)         # (-1, 7, 7, 8)
            x = MaxPooling2D((2, 2), padding='same')(x)                         # (-1, 4, 4, 8)
            x = Flatten()(x)                                                    # (-1, 128)
            x = Dense(32, activation='relu')(x)                                 # (-1, 32)
        else:
            x = encoder
            inputs = inputlayer

        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        z = Lambda(_sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])  # (-1, 2)

        complete_encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        complete_encoder.summary()

        return complete_encoder, inputs, z_mean, z_log_var

    def _build_decoder(self, input_shape, latent_dim, decoder=None, decoderinputs=None):
        """
        Builds the decoder portion of the model and returns it.
        """
        if decoder is None:
            intermediate_dim = (4, 4, 8)  # hard-coded to MNIST
            latent_inputs = Input(shape=(latent_dim,), name='z_sampling')        # (-1, 2)
            x = Dense(32, activation='relu')(latent_inputs)                      # (-1, 32)
            x = Dense(np.product(intermediate_dim), activation='relu')(x)        # (-1, 128)
            x = Reshape(target_shape=intermediate_dim)(x)                        # (-1, 4, 4, 8)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (-1, 4, 4, 8)
            x = UpSampling2D((2, 2))(x)                                          # (-1, 8, 8, 8)
            x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (-1, 8, 8, 8)
            x = UpSampling2D((2, 2))(x)                                          # (-1, 16, 16, 8)
            x = Conv2D(16, (3, 3), activation='relu')(x)                         # (-1, 14, 14, 16)
            x = UpSampling2D((2, 2))(x)                                          # (-1, 28, 28, 16)
            outputs = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # (-1, 28, 28, 1)
        else:
            latent_inputs = decoderinputs
            outputs = decoder

        # instantiate decoder model
        completed_decoder = Model(latent_inputs, outputs, name='decoder')
        completed_decoder.summary()

        return completed_decoder

if __name__ == '__main__':
    # If this module is run as a script, do a smoke test
    # Further tests are present in the tests directory of this project

    import matplotlib.pyplot as plt

    # Put together argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--latentdim", default=2, type=int, help="Dimensionality of the latent vector space")
    parser.add_argument("-b", "--batchsize", default=128, type=int, help="Number of input vectors to batch as a single training instance")
    parser.add_argument("-e", "--nepochs", default=50, type=int, help="Number of passes through the whole dataset for training")
    parser.add_argument("-o", "--optimizer", default="adam", type=str, help="Optimizer to use for training the VAE")
    parser.add_argument("-l", "--loss", default="mse", choices=["mse"], help="The loss function to use along with KL divergence")
    parser.add_argument("-s", "--save", default="vae_mnist.h5", type=str, help="File path to save the weights at. If empty, we will not save the weights.")
    parser.add_argument("-w", "--weights", type=str, help="Load the given file as weights and work in inference mode, rather than training first")
    args = parser.parse_args()

    # Validate the inputs
    if args.weights and not os.path.isfile(args.weights):
        print("Could not find weight file", args.weights)
        exit(1)

    # Get the MNIST data, along with some parameters about it
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = (28, 28, 1)

    # Reshape into the appropriate shapes and types
    x_train = np.reshape(x_train, [-1, *original_dim])
    x_test = np.reshape(x_test, [-1, *original_dim])
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Create the VAE
    vae = VariationalAutoEncoder(input_shape=original_dim,
                                 latent_dim=args.latentdim,
                                 optimizer=args.optimizer,
                                 loss=args.loss)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train, x_train, epochs=args.nepochs, batch_size=args.batchsize, validation_data=(x_test, x_test))
        if args.save:
            vae.save_weights(args.save)

    if args.latentdim == 2:
        # Do some inference and plot it for human consumption

        # display a 30x30 2D manifold of digits
        n = 30
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))

        # linearly spaced coordinates corresponding to the 2D plot
        # of digit classes in the latent space
        grid_x = np.linspace(-4, 4, n)
        grid_y = np.linspace(-4, 4, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = vae.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                    j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range + 1
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap='Greys_r')
        plt.show()
    else:
        print("To see decoded samples, run with --latentdim 2")
