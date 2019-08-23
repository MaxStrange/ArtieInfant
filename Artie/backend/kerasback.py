"""
This is the Keras back-end API.
"""
import keras
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from internals.vae import vae                           # pylint: disable=locally-disabled, import-error
from internals.vae import ae as vanilla                 # pylint: disable=locally-disabled, import-error
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import preprocessing


class VaeVisualizer(keras.callbacks.Callback):
    """
    Callback to pass into the VAE model when building it. This callback generates a before and after spectrogram
    sampled from a random batch after each epoch.
    """
    def __init__(self):
        # TODO: This is currently caching ALL of the data that we train on. We should not do that.
        super(VaeVisualizer, self).__init__()
        self.targets = []
        self.outputs = []
        self.inputs = []

        self.var_y_true = tf.Variable(0.0, validate_shape=False)
        self.var_y_pred = tf.Variable(0.0, validate_shape=False)
        self.var_x = tf.Variable(0.0, validate_shape=False)

    def on_batch_end(self, batch, logs=None):
        self.targets.append(K.eval(self.var_y_true))
        self.outputs.append(K.eval(self.var_y_pred))
        self.inputs.append(K.eval(self.var_x))

    def on_epoch_end(self, epoch, logs=None):
        # Get a random number to determine which batch to get
        batchidx = random.randint(0, len(self.inputs) - 1)

        # Use that number as an index into the batches from this epoch
        input_spec_batch = self.inputs[batchidx]

        # Get another random number
        idx = random.randint(0, input_spec_batch.shape[0] - 1)

        # Use that number as an index into the batch to get a random spectrogram
        input_spec = input_spec_batch[idx]

        # Get the times and frequencies (not the real ones, just some dummy values that we can feed into matplotlib)
        times = np.arange(0, input_spec.shape[1])
        freqs = np.arange(0, input_spec.shape[0])

        # Reshape the input spectrogram into the right shape for matplotlib
        inp = np.reshape(input_spec, (len(freqs), len(times)))

        # Plot the input spectrogram on the left (also modify the amplitudes to make them more visible)
        plt.subplot(121)
        plt.title("Input (batch, idx): {}".format((batchidx, idx)))
        plt.pcolormesh(times, freqs, inp)

        # Get the corresponding output spectrogram
        output_spec = self.outputs[batchidx][idx]

        # Reshape the output spectrogram into the right shape for matplotlib
        outp = np.reshape(output_spec, (len(freqs), len(times)))

        # Plot it on the right
        plt.subplot(122)
        plt.title("Output (batch, idx): {}".format((batchidx, idx)))
        plt.pcolormesh(times, freqs, outp)

        # Show the user
        plt.show()


def seed(s: int):
    """
    Set the random seed.
    """
    tf.set_random_seed(s)

def build_autoencoder1(input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds and returns a vanilla autoencoder for 241x20x1.
    """
    inputs = Input(shape=input_shape, name="encoder-input")                 # (-1, 241, 20, 1)
    x = Conv2D(128, (8, 2), strides=(2, 1), activation='relu', padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 2), strides=(2, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (8, 2), strides=(2, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (9, 2), strides=(2, 2), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
    x = Flatten()(x)
    encoder = Dense(128, activation='relu')(x)

    decoderinputs = Input(shape=(latent_dim,), name='decoder-input')
    x = Dense(128, activation='relu')(decoderinputs)
    x = Reshape(target_shape=(4, 4, 8))(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3 ,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3 ,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3 ,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3 ,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 4), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((1, 2))(x)
    x = Conv2D(32, (8, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoder = Conv2D(1, (2, 1), activation='relu', padding='valid')(x)

    autoencoder = vanilla.AutoEncoder(input_shape, latent_dim, optimizer, loss, encoder, decoder, inputs, decoderinputs)
    return autoencoder

def build_autoencoder2(input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds and returns an 81x18x1 autoencoder.
    """
    inputs = Input(shape=input_shape, name="encoder-input")                 # (-1, 81, 18, 1)
    x = Conv2D(128, (8, 3), strides=(2, 1), activation='relu', padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (6, 3), strides=(1, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    encoder = Dense(128, activation='relu')(x)

    decoderinputs = Input(shape=(latent_dim,), name='decoder-input')
    x = Dense(128, activation='relu')(decoderinputs)
    x = Reshape(target_shape=(4, 4, 8))(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (6, 3), activation='relu', padding='valid')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 1), activation='relu', padding='valid')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (8, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    decoder = Conv2D(1, (2, 1), activation='relu', padding='valid')(x)

    autoencoder = vanilla.AutoEncoder(input_shape, latent_dim, optimizer, loss, encoder, decoder, inputs, decoderinputs)
    return autoencoder


def build_variational_autoencoder1(input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds a variational autoencoder and returns it.
    """
    # Encoder model
    inputs = Input(shape=input_shape, name="encoder-input")                 # (-1, 241, 20, 1)
    x = Conv2D(128, (8, 2), strides=(2, 1), activation='relu', padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 2), strides=(2, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (8, 2), strides=(2, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (9, 2), strides=(2, 2), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
    x = Flatten()(x)
    encoder = Dense(128, activation='relu')(x)

    # Decoder model
    decoderinputs = Input(shape=(latent_dim,), name='decoder-input')
    x = Dense(128, activation='relu')(decoderinputs)
    x = Reshape(target_shape=(4, 4, 8))(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3 ,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3 ,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3 ,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3 ,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 4), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((1, 2))(x)
    x = Conv2D(32, (8, 2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    decoder = Conv2D(1, (2, 1), activation='relu', padding='valid')(x)

    autoencoder = vae.VariationalAutoEncoder(input_shape, latent_dim, optimizer, loss,
                                                kl_loss_prop=kl_loss_prop, recon_loss_prop=recon_loss_prop, std_loss_prop=std_loss_prop,
                                                encoder=encoder, decoder=decoder, inputlayer=inputs,
                                                decoderinputlayer=decoderinputs, tbdir=tbdir)
    return autoencoder

def build_variational_autoencoder2(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    """
    # Encoder model
    inputs = Input(shape=input_shape, name="encoder-input")                 # (-1, 81, 18, 1)
    x = Conv2D(128, (8, 3), strides=(2, 1), activation='relu', padding='valid')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (6, 3), strides=(1, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (4, 4), strides=(1, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), strides=(1, 1), activation='relu', padding='valid')(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    encoder = Dense(128, activation='relu')(x)

    decoderinputs = Input(shape=(latent_dim,), name='decoder-input')
    x = Dense(128, activation='relu')(decoderinputs)
    x = Reshape(target_shape=(4, 4, 8))(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (6, 3), activation='relu', padding='valid')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (6, 3), strides=(2, 1), activation='relu', padding='valid')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 1), activation='relu', padding='valid')(x)
    x = UpSampling2D((2, 1))(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (8, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (8, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 1), activation='relu', padding='valid')(x)
    x = BatchNormalization()(x)
    decoder = Conv2D(1, (2, 1), activation='relu', padding='valid')(x)

    autoencoder = vae.VariationalAutoEncoder(input_shape, latent_dim, optimizer, loss,
                                                kl_loss_prop=kl_loss_prop, recon_loss_prop=recon_loss_prop, std_loss_prop=std_loss_prop,
                                                encoder=encoder, decoder=decoder, inputlayer=inputs,
                                                decoderinputlayer=decoderinputs, tbdir=tbdir)
    return autoencoder

def train_autoencoder(autoencoder, visualize: bool, root: str, steps_per_epoch: int, imshapes: [int], batchsize: int, nworkers: int, test: str, nepochs: int):
    """
    Train the autoencoder.
    """
    # Deal with the visualization stuff if we are visualizing
    if visualize:
        vaeviz = VaeVisualizer()
        model = autoencoder._vae
        fetches = [tf.assign(vaeviz.var_y_true, model.targets[0], validate_shape=False),
                   tf.assign(vaeviz.var_y_pred, model.outputs[0], validate_shape=False),
                   tf.assign(vaeviz.var_x, model.inputs[0], validate_shape=False)]
        model._function_kwargs = {'fetches': fetches}
        callbacks = [vaeviz]
    else:
        callbacks = []

    logging.info("Loading images from {}".format(root))
    nsteps_per_validation = steps_per_epoch
    imreader = preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    print("Creating datagen...")
    datagen = imreader.flow_from_directory(root,
                                            target_size=imshapes,
                                            color_mode='grayscale',
                                            classes=None,
                                            class_mode='input',
                                            batch_size=batchsize,
                                            shuffle=True,
                                            save_to_dir=None,
                                            save_format='png')
    print("Creating testgen...")
    testreader = preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    testgen = testreader.flow_from_directory(test,#root,
                                            target_size=imshapes,
                                            color_mode='grayscale',
                                            classes=None,
                                            class_mode='input',
                                            batch_size=batchsize,
                                            shuffle=True,
                                            save_to_dir=None,
                                            save_format='png')
    print("Training...")
    autoencoder.fit_generator(datagen,
                              batchsize,
                              epochs=nepochs,
                              save_models=True,
                              steps_per_epoch=steps_per_epoch,
                              use_multiprocessing=False,
                              workers=nworkers,
                              callbacks=callbacks,
                              validation_data=testgen,
                              validation_steps=nsteps_per_validation)
