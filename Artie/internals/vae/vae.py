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

from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, MaxPooling2D, Flatten, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

class VariationalAutoEncoder:
    """
    A class to represent an encoder and decoder such that the encoder
    is constrained to learning an embedding of the input vectors
    that conforms to a particular distribution.
    """
    def __init__(self, input_shape, intermediate_dim, latent_dim, optimizer, loss):
        """
        :param input_shape:         The shape of the input data.
        :param intermediate_dim:    The intermediate layer length.
        :param latent_dim:          The dimensionality of the latent vector space.
        :param optimizer:           String representation of the optimizer.
        :param loss:                String representation of the loss function.
        """
        self._encoder, self._inputs, z_mean, z_log_var = self._build_encoder(input_shape, intermediate_dim, latent_dim)
        self._decoder = self._build_decoder(input_shape, intermediate_dim, latent_dim)
        self._outputs = self._decoder(self._encoder(self._inputs)[2])
        self._vae = Model(self._inputs, self._outputs, name='vae_mlp')
        flattened_input_shape = (np.product(np.array(input_shape)),)
        reconstruction_loss = self._build_loss(loss, flattened_input_shape)
        kl_loss = self._build_kl_loss(z_log_var, z_mean)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self._vae.add_loss(vae_loss)
        self._vae.compile(optimizer=optimizer)
        self._vae.summary()

    def predict(self, *args, **kwargs):
        """
        Do inference with the VAE. See the Keras predict() method's documentation: https://keras.io/models/model/#predict
        """
        return self._decoder.predict(*args, **kwargs)

    def fit(self, *args, **kwargs):
        """
        Train the VAE on the given data. See the Keras fit() method's documentation: https://keras.io/models/model/#fit
        """
        return self._vae.fit(*args, **kwargs)

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

    def _build_kl_loss(self, z_log_var, z_mean):
        """
        Builds and returns the KL loss function.
        """
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return kl_loss

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

    def _build_encoder(self, input_shape, intermediate_dim, latent_dim):
        """
        Builds the encoder portion of the model and returns it.
        """
        inputs = Input(shape=input_shape, name="encoder_input")             # (-1, 28, 28, 1)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)   # (-1, 28, 28, 16)
        x = MaxPooling2D((2, 2), padding='same')(x)                         # (-1, 14, 14, 16)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)         # (-1, 14, 14, 8)
        x = MaxPooling2D((2, 2), padding='same')(x)                         # (-1, 7, 7, 8)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)         # (-1, 7, 7, 8)
        x = MaxPooling2D((2, 2), padding='same')(x)                         # (-1, 4, 4, 8)
        x = Flatten()(x)                                                    # (-1, 128)
        x = Dense(32, activation='relu')(x)                                 # (-1, 32)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])  # (-1, 2)

        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder.summary()

        return encoder, inputs, z_mean, z_log_var

    def _build_decoder(self, input_shape, intermediate_dim, latent_dim):
        """
        Builds the decoder portion of the model and returns it.
        """
        intermediate_dim = (4, 4, 8)  # TODO: This is hard-coded to MNIST and is whatever the last shape is before the Dense portion of then encoder

        # build decoder model
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

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()

        return decoder

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

if __name__ == '__main__':
    # If this module is run as a script, do a smoke test
    # Further tests are present in the tests directory of this project

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
                                 intermediate_dim=512,
                                 latent_dim=args.latentdim,
                                 optimizer=args.optimizer,
                                 loss=args.loss)

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train, epochs=args.nepochs, batch_size=args.batchsize, validation_data=(x_test, None))
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
