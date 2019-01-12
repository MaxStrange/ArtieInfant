import audiosegment
import sys

import math
import matplotlib.pyplot as plt
import numpy as np

def get_reward(seg):
    arr = seg.to_numpy_array()
    assert len(arr) > 0
    squares = np.square(arr)
    assert len(squares) == len(arr)
    sum_of_squares = np.sum(squares[squares >= 0], axis=0)
    assert sum_of_squares >= 0.0, "Len: {}, Sum of squares: {}".format(len(arr), np.sum(squares, axis=0))
    mean_square = sum_of_squares / len(arr)
    assert mean_square > 0.0
    rms = np.sqrt(mean_square)
    rew = rms
    if math.isnan(rew):
        rew = 0.0
    rew /= 100.0  # Get it into a more reasonable domain
    return rew


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need a path to a WAV file.")
        exit(1)

    seg = audiosegment.from_file(sys.argv[1])
    rew = get_reward(seg)
    print("Reward for this WAV file:", rew)

    plt.plot(seg.to_numpy_array())
    plt.show()
