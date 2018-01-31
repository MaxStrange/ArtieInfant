"""
"""
import matplotlib.pyplot as plt
import os
import sys

base = "/home/max/git_repos/ArtieInfant/scratch/mlpvad/"

def plot_one(fpath, i):
    with open(fpath) as f:
        vals = [float(val.strip()) for val in f]
        plt.plot(vals)

if __name__ == "__main__":
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        try:
            plot_one(os.sep.join([base, "BATCH_0_ClassLabels.NO_VOICE_" + str(i) + ".csv"]), i)
            plt.title("NO")
        except FileNotFoundError:
            plot_one(os.sep.join([base, "BATCH_0_ClassLabels.VOICE_" + str(i) + ".csv"]), i)
            plt.title("YES")
    plt.tight_layout()
    plt.show()

