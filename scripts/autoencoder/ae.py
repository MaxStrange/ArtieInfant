"""
See README.md
"""
import keras

from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, MaxPooling2D, Flatten, Reshape, BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

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


if __name__ == "__main__":
    convolutional = True

    ##### SPECT #####
    # Load the dataset
    images = []
    for root, _, fnames in os.walk("/home/maxst/repos/ArtieInfant/scripts/test_vae/example_spectrograms/"):
        for fname in fnames:
            fpath = os.path.join(root, fname)
            if os.path.splitext(fpath)[-1].lower() == ".png":
                im = imageio.imread(fpath)
                images.append(im)
    x_train = np.array(images)
    imshape = x_train[0].shape
    imlength = np.prod(imshape)
    print("XTRAIN:", x_train.shape)
    print("IMSHAPE:", imshape)
    print("IMLENGTH:", imlength)

    if convolutional:
        x_train = np.expand_dims(x_train, -1)  # Add channel image
        ae = cnn_ae(x_train[0].shape)
        ae.compile('adadelta', loss='mse')
        ae.fit(x_train, x_train, epochs=200, batch_size=4)
    else:
        x_train = np.reshape(x_train, (-1, imlength))
        x_train = np.expand_dims(x_train[0,:], 0)
        ae = vanilla_ae(imlength)
        ae.compile('adadelta', loss='mse')
        ae.fit(x_train, x_train, epochs=200, batch_size=4)

    outputs = ae.predict(np.expand_dims(x_train[0,:], 0))
    inputs = np.reshape(x_train[0,:], imshape)
    outputs = np.reshape(outputs, imshape)
    print("Input shape:", inputs.shape)
    print("Output shape:", outputs.shape)

    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(inputs)
    plt.subplot(1, 2, 2)
    plt.title("Output")
    plt.imshow(outputs)
    plt.show()

    ##### MNIST #####
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #image_size = x_train.shape[1]

    #if convolutional:
    #    original_dim = (28, 28, 1)
    #    x_train = np.reshape(x_train, [-1, *original_dim])
    #    x_test = np.reshape(x_test, [-1, *original_dim])
    #else:
    #    x_train = np.reshape(x_train, (-1, image_size * image_size))
    #    x_test = np.reshape(x_train, (-1, image_size * image_size))

    #x_train = x_train.astype('float32') / 255.0
    #x_test = x_test.astype('float32') / 255.0

    #if convolutional:
    #    pass
    #else:
    #    ae = vanilla_ae(image_size * image_size)

    #ae.compile('adadelta', loss='mse')
    #ae.fit(x_train, x_train, epochs=1, batch_size=32, validation_data=(x_test, x_test))
    #outputs = ae.predict(np.expand_dims(x_test[0, :], 0))

    #inputs = np.reshape(x_train[0,:], (image_size, image_size))
    #outputs = np.reshape(outputs, (image_size, image_size))
    #print("Input shape:", inputs.shape)
    #print("Output shape:", outputs.shape)

    #plt.subplot(1, 2, 1)
    #plt.title("Input")
    #plt.imshow(inputs)
    #plt.subplot(1, 2, 2)
    #plt.title("Output")
    #plt.imshow(outputs)
    #plt.show()
