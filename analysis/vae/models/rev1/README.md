# Rev 1

This directory contains VAE models corresponding to this architecture:

```python
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
```
