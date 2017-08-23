"""
This script is used to train a given model.
"""
import keras
import os
import sys

# Parameters
MODEL_NAME = "version_0.1.0"
WINDOW_WIDTH_MS = 30  # How many MS of audio to feed into the MLP at a time
SAMPLING_RATE_HZ = 32000  # Sample the audio at this rate
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
    model.add(keras.layers.Dense(1024, input_dim=samples_per_window))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dropout(rate=0.33))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(rate=0.33))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    model.compile(optimizer="adagrad", loss="binary_crossentropy", metrics=["accuracy"])

    checkpointer = keras.callbacks.ModelCheckpoint(model_dir_path)
    model.fit(data, one_hot_labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=checkpointer)
