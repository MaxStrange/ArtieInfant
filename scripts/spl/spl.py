import audiosegment
import sys

import math
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need a path to a WAV file.")
        exit(1)

    seg = audiosegment.from_file(sys.argv[1])
    print("Reward for this WAV file:", seg.rms)

    plt.plot(seg.to_numpy_array())
    plt.show()
