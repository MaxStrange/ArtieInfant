"""
See README.md
"""
import keras
from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, MaxPooling2D, Flatten, Reshape, BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import sys
import vae

def vanilla_ae(input_length):
    """
    Returns an MLP autoencoder model.
    """
    input_shape = (input_length,)
    inputs = Input(shape=input_shape, name="encoder-input")
    x = Dense(5124)(inputs)
    x = BatchNormalization()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Dense(512)(x)
    x = Dense(32, name="embedding-layer")(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Dense(5124)(x)
    x = Dense(input_length)(x)
    return Model(inputs=inputs, outputs=x)

def cnn_vae(input_shape, latent_dim, optimizer, loss, tbdir=None):
    """
    Returns a CNN variational autoencoder model.
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

    ae = vae.VariationalAutoEncoder(input_shape, latent_dim, optimizer, loss,
                                encoder=encoder, decoder=decoder, inputlayer=inputs,
                                decoderinputlayer=decoderinputs, tbdir=tbdir)
    return ae

def cnn_ae(input_shape):
    """
    Returns a CNN autoencoder model.
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
    x = Dense(32, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
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
    x = Conv2D(1, (2, 1), activation='relu', padding='valid')(x)

    m = Model(inputs=inputs, outputs=x)
    m.summary()

    return m

def load_data(datadir, maxnum=2000):
    """
    Loads all the spectrograms found in datadir up to maxnum.
    """
    n = 0
    images = []
    for root, _, fnames in os.walk(sys.argv[1]):
        for fname in fnames:
            fpath = os.path.join(root, fname)
            if os.path.splitext(fpath)[-1].lower() == ".png":
                im = imageio.imread(fpath)
                images.append(im)
                n += 1
                if n >= maxnum:
                    break
    x_train = np.array(images)
    return x_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batchsize", default=32, type=int)
    parser.add_argument("-e", "--nepochs", default=50, type=int)
    parser.add_argument("-f", "--fully-connected", action="store_true")
    parser.add_argument("-l", "--loss", type=str, default="mse")
    parser.add_argument("-o", "--optimizer", type=str, default="adadelta")
    parser.add_argument("-v", "--vae", action="store_true")
    parser.add_argument("-t", "--tbdir", default=None, help="Optional directory for tensor board files")
    parser.add_argument("datadir", type=str, help="Location of the data to train on")
    args = parser.parse_args()

    if args.fully_connected:
        convolutional = False
    else:
        convolutional = True

    x_train = load_data(args.datadir)
    assert x_train.shape[0] > 0, "Data directory was empty of spectrograms"
    imshape = x_train[0].shape
    imlength = np.prod(imshape)

    if convolutional:
        x_train = np.expand_dims(x_train, -1)  # Add channel image
        x_train = x_train / 255.0
        if args.vae:
            latentdim = 2
            ae = cnn_vae(x_train[0].shape, latentdim, args.optimizer, args.loss, args.tbdir)
        else:
            ae = cnn_ae(x_train[0].shape)
            ae.compile(args.optimizer, loss=args.loss)
        ae.fit(x_train, x_train, epochs=args.nepochs, batch_size=args.batchsize)
    else:
        x_train = np.reshape(x_train, (-1, imlength))
        x_train = np.expand_dims(x_train[0,:], 0)
        ae = vanilla_ae(imlength)
        ae.compile('adadelta', loss='mse')
        ae.fit(x_train, x_train, epochs=200, batch_size=4)

    if args.vae:
        ae.save_weights("models/VAE.h5")

    for i in range(min(5, x_train.shape[0])):
        if args.vae:
            outputs = ae.encode_decode(np.expand_dims(x_train[i,:], 0))
            outputs *= 255.0
        else:
            outputs = ae.predict(np.expand_dims(x_train[i,:], 0))
        inputs = np.reshape(x_train[i,:], imshape)
        outputs = np.reshape(outputs, imshape)

        plt.subplot(2, 5, i + 1)
        plt.title("Input {}".format(i))
        plt.imshow(inputs)
        plt.subplot(2, 5, i + 5 + 1)
        plt.title("Output {}".format(i))
        plt.imshow(outputs)
    plt.show()
