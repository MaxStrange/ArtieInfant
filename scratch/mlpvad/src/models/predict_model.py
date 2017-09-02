"""
This script runs the given model over all the test directories in the
given path and reports the results for visualisation.
"""
import keras
import keras.models
import numpy as np
import os
import src.features.build_features as build_features
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Please go to the top-level directory and run this through `make test`.")
        exit(1)

    model_path = sys.argv[1]
    test_root = sys.argv[2]

    if not os.path.isfile(model_path):
        print("File at path:", model_path, "either does not exist or is not a file.")
        exit(1)

    if not os.path.isdir(test_root):
        print("Root directory at path:", test_root, "either does not exist or is not a directory.")
        exit(1)

    test_dirs = [dpath for dpath, __, _fname in os.walk(test_root) if "test_split" in dpath.split(os.sep)[-1]]
    if not test_dirs:
        print("No test directories detected anywhere inside", test_root)
        exit(1)

    model = keras.models.load_model(model_path)

    prediction_results = {}
    for test_dir in test_dirs:
        prediction_results[test_dir] = []
        for dpath, __, fpaths in os.walk(test_dir):
            for fpath in fpaths:
                if fpath.lower().endswith("wav"):
                    full_fpath = os.sep.join([dpath, fpath])
                    for vector, label in build_features.generate_vectors_and_labels_from_file(full_fpath):
                        result = model.predict(np.array([vector]), batch_size=1)
                        print("Got:", result, "expected:", label)
                        prediction_results[test_dir].append((result, label))

    # Visualize the results using whatever means (confusion matrix, accuracy, precision, recall, F0 score)
    # TODO
    # prediction results is a data structure that looks like this:
    # {test_dir : [(guess, label), (guess, label), etc.], another_test_dir : [etc.], etc.}
