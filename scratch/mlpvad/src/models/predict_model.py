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
from tqdm import tqdm

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
        print("Checking:", test_dir)
        prediction_results[test_dir] = []
        for dpath, __, fpaths in os.walk(test_dir):
            print("  |-> Checking", dpath)
            for fpath in tqdm(fpaths):
                if fpath.lower().endswith("wav"):
                    full_fpath = os.sep.join([dpath, fpath])
                    for vector, label in build_features.generate_vectors_and_labels_from_file(full_fpath):
                        result = model.predict(np.array([vector]), batch_size=1)
                        result = int(round(result[0][0]))
                        label = int(label)
                        prediction_results[test_dir].append((result, label))

    correct_nos = 0
    correct_yes = 0
    no_should_be_yes = 0
    yes_should_be_no = 0
    for test_dir, results_list in prediction_results.items():
        for guess, label in results_list:
            if guess == 0 and label == 0:
                correct_nos += 1
            elif guess == 1 and label == 1:
                correct_yes += 1
            elif guess == 0 and label == 1:
                no_should_be_yes += 1
            else:
                yes_should_be_no += 1

    def percent(x):
        total = correct_nos + correct_yes + no_should_be_yes + yes_should_be_no
        return (x / total) * 100.0

    print("Correct nos:", correct_nos, "    ", percent(correct_nos), "% of total guesses")
    print("Correct yes:", correct_yes, "    ", percent(correct_yes), "% of total guesses")
    print("False negatives:", no_should_be_yes, "    ", percent(no_should_be_yes), "% of total guesses")
    print("False positives:", yes_should_be_no, "    ", percent(yes_should_be_no), "% of total guesses")
