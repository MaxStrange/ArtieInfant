"""
Online logging script to be run concurrently with make train.
"""
import functools
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

import random

FILE_PATH = "log.csv"

class Plotter:
    def __init__(self, x1, y1, x2, y2, getter):
        self._fig = plt.figure()
        self._ax1 = self._fig.add_subplot(1, 2, 1)
        self._ax2 = self._fig.add_subplot(1, 2, 2)

        self._data_y1 = y1
        self._data_x1 = x1
        self._data_y2 = y2
        self._data_x2 = x2

        self._data_getter = getter

    def animate(self):
        ani = animation.FuncAnimation(self._fig, self._draw, interval=1)
        plt.show()

    def _update_data(self):
        new_y1, new_y2 = self._data_getter.get()
        self._data_y1.extend(new_y1)
        self._data_y2.extend(new_y2)
        return new_y1, new_y2

    def _draw(self, frame):
        new_y1s, new_y2s = self._update_data()

        if new_y1s:
            new_x1s = range(len(self._data_x1), len(self._data_x1) + len(new_y1s))
            new_x2s = range(len(self._data_x2), len(self._data_x2) + len(new_y2s))
            self._data_x1.extend(new_x1s)
            self._data_x2.extend(new_x2s)

        self._ax1.clear()
        self._ax1.plot(self._data_x1, self._data_y1)
        self._ax1.set_title("Loss")
        self._ax2.clear()
        self._ax2.plot(self._data_x2, self._data_y2)
        self._ax2.set_title("Accuracy")

class Getter:
    """
    Class that provides a 'get' function for retrieving one data point at a time.
    """
    def __init__(self, fpath):
        """
        :param fpath: The path to the file to read from.
        :param dataindex: A parameter that indicates which index in the tuple of data items on a line to get.
        """
        self.fpath = fpath
        self.epoch_num = 0

    def get(self):
        epoch_str = lambda epnum : "----- " + str(epnum) + " -----"
        with open(self.fpath) as f:
            # Take only the lines after the most recent epoch
            lines = [line.strip() for line in itertools.dropwhile(lambda x: x.strip() != epoch_str(self.epoch_num), f)]
            lines = [line.strip() for line in itertools.takewhile(lambda x: x.strip() != epoch_str(self.epoch_num + 1), lines)]
            lines = [line for line in lines if line.strip() != "" and not line.startswith('-')]
            tups = [line.split(',') for line in lines]
            data_x1 = [float(tup[0].strip()) for tup in tups]
            data_x2 = [float(tup[1].strip()) for tup in tups]
        if data_x1:
            self.epoch_num += 1
        return data_x1, data_x2

if __name__ == "__main__":
    acc = 0
    loss = 1
    g = Getter(FILE_PATH)
    plotter = Plotter([0, 1], [0, 1], [0, 1], [0, 1], g)
    plotter.animate()

