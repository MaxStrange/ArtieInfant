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
    Returns an MLP autoencoder Sequential model.
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

if __name__ == "__main__":
    convolutional = False

    ##### SPECT #####
    # Load the dataset
    images = []
    for root, _, fnames in os.walk("/home/max/repos/ArtieInfant/scripts/test_vae/example_spectrograms/"):
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

    if not convolutional:
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
