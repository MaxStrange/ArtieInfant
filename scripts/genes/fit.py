"""
Script I am using to tune/debug the fitness function for phase 1.
"""
import audiosegment as asg
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot(seg, title):
    plt.title(title)
    plt.plot(seg)
    plt.show()

if __name__ == "__main__":
    seg = asg.from_file(sys.argv[1])

    ours = seg.to_numpy_array().astype(float)

    most_neg_val = min(ours)
    ours += abs(most_neg_val)

    assert sum(ours[ours < 0]) == 0

    if max(ours) != min(ours):
        ours /= max(ours) - min(ours)

    other = asg.from_file(sys.argv[2])
    other = other.to_numpy_array().astype(float)
    other += abs(min(other))
    if max(other) != min(other):
        other /= max(other) - min(other)

    plot(ours, "Ours")
    plot(other, "Other")

    xcor = np.correlate(ours, other, mode='full')
    plot(xcor, "XCor")

    print("Reward would be:", max(xcor))
