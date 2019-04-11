"""
Verifies if all the spectrograms in a directory are the same dimensionality.
Prints out any that are not the expected dimensionality.
"""
import imageio
import os
import sys
import tqdm

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python {} <path-to-dir> <remove: y/n>".format(sys.argv[0]))
        exit(1)
    elif not os.path.isdir(sys.argv[1]):
        print("{} is not a valid directory.".format(sys.argv[1]))
        exit(2)

    remove = sys.argv[2].strip().lower() == 'y'

    targetdir = sys.argv[1]
    fpaths = [os.path.join(targetdir, fname) for fname in os.listdir(targetdir)]
    for fpath in tqdm.tqdm(fpaths):
        spec = imageio.imread(fpath)
        if spec.shape != (241, 20):
            #print("{} has shape {}".format(fpath, spec.shape))
            if remove:
                os.remove(fpath)
