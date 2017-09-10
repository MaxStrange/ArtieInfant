"""
This script is used to train a given model.
"""
import collections
import keras
import keras.backend as K
import os
import sys
import src.features.build_features as build_features
import src.models.metrics as metrics

# Parameters
WINDOW_WIDTH_MS = 30  # How many MS of audio to feed into the MLP at a time
SAMPLING_RATE_HZ = 32000  # Sample the audio at this rate
NUM_CHANNELS = 1  # The number of channels in the audio
NUM_EPOCHS = 10000
BATCH_SIZE = 32
LOG_FILE = "log.csv"

class GraphMetrics(keras.callbacks.Callback):
    def __init__(self, metrics):
        """
        """
        self.batch_num = 0
        self.metrics_logs = collections.OrderedDict({"loss" : []})
        if "accuracy" in metrics:
            self.metrics_logs["acc"] = []
        for m in metrics:
            if hasattr(m, "__name__"):
                self.metrics_logs[m.__name__] = []

        # Make the file
        with open(LOG_FILE, 'w') as f:
            f.write(", ".join([metric for metric in self.metrics_logs.keys()]))

    def on_batch_end(self, batch, logs={}):
        for metric in self.metrics_logs.keys():
            self.metrics_logs[metric].append(float(logs.get(metric)))
        with open(LOG_FILE, 'a') as f:
            f.write(os.linesep)
            f.write("----- " + str(self.batch_num) + " -----" + os.linesep)

            # This is pretty crappy code. There must be a way to do this with zip, but I
            # am too busy to think about it right now.
            metrics = [m for m in self.metrics_logs.values()]
            for i in range(len(metrics[0])):
                values = [m[i] for m in metrics]
                f.write(", ".join([str(v) for v in values]))
                f.write(os.linesep)
            for ls in self.metrics_logs.values():
                ls.clear()

        self.batch_num += 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: Go to the top level directory and run `make train`")
        exit(1)

    model_dir_path = sys.argv[1]
    data_dir_path = sys.argv[2]
    samples_per_window = int(WINDOW_WIDTH_MS * SAMPLING_RATE_HZ / 1000)
    print("Input dimension:", samples_per_window)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(4024, input_dim=samples_per_window // 2 + 1))  # Takes the abs of the FFT output
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1012))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(512))
    model.add(keras.layers.Dropout(0.10))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    metrics = ["accuracy", metrics.fscore]
    model.compile(optimizer="nadam", loss="binary_crossentropy", metrics=metrics)

    kwargs = {"samples_per_vector": samples_per_window,
              "batch_size": BATCH_SIZE,
              "sampling_frequency_hz": SAMPLING_RATE_HZ,
              "channels": NUM_CHANNELS,
              "ignore": ["debug_NO", "debug_VO", "test_split"],
              "include": None,
             }
    data_generator = build_features.generate_data(data_dir_path, **kwargs)
    steps_per_epoch = build_features.calculate_steps_per_epoch(data_dir_path, **kwargs)
    checkpointer = keras.callbacks.ModelCheckpoint(model_dir_path, period=0.2)
    metrics_grapher = GraphMetrics(metrics)
    model.fit_generator(data_generator, steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS,
                        callbacks=[checkpointer, metrics_grapher], verbose=1)

