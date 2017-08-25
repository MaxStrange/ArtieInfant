"""
This script is used to train a given model.
"""
import keras
import os
import sys
import src.features.build_features as build_features

# Parameters
MODEL_NAME = "version_0.1.0"
WINDOW_WIDTH_MS = 160  # How many MS of audio to feed into the MLP at a time
SAMPLING_RATE_HZ = 32000  # Sample the audio at this rate
NUM_CHANNELS = 1  # The number of channels in the audio
NUM_EPOCHS = 10
BATCH_SIZE = 32

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: Go to the top level directory and run `make train`")
        exit(1)

    model_dir_path = sys.argv[1]
    data_dir_path = sys.argv[2]
    samples_per_window = int(WINDOW_WIDTH_MS * SAMPLING_RATE_HZ / 1000)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(2048, input_dim=samples_per_window))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.Dropout(rate=0.33))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dropout(rate=0.33))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    model.compile(optimizer="adagrad", loss="binary_crossentropy", metrics=["accuracy"])

    data_generator = build_features.generate_data(data_dir_path, samples_per_vector=samples_per_window, batch_size=BATCH_SIZE, sampling_frequency_hz=SAMPLING_RATE_HZ)
    steps_per_epoch = build_features.calculate_steps_per_epoch(data_dir_path, samples_per_vector=samples_per_window, batch_size=BATCH_SIZE, sampling_frequency_hz=SAMPLING_RATE_HZ, channels=NUM_CHANNELS)
    checkpointer = keras.callbacks.ModelCheckpoint(model_dir_path)
    model.fit_generator(data_generator, steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS, callbacks=[checkpointer], verbose=1)

