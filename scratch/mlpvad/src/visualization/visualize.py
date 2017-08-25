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
    def __init__(self, x, y, getter):
        self._fig = plt.figure()
        self._ax1 = self._fig.add_subplot(1, 1, 1)

        self._data_y = y
        self._data_x = x

        self._data_getter = getter

    def animate(self):
        ani = animation.FuncAnimation(self._fig, self._draw, interval=500)
        plt.show()

    def _update_data(self):
        new_vals = self._data_getter.get()
        self._data_y.extend(new_vals)
        return new_vals

    def _draw(self, frame):
        got_data = self._update_data()

        if got_data:
            new_xs = range(len(self._data_x), len(self._data_x) + len(got_data))
            self._data_x.extend(new_xs)

        self._ax1.clear()
        self._ax1.plot(self._data_x, self._data_y)

class Getter:
    """
    Class that provides a 'get' function for retrieving one data point at a time.
    """
    def __init__(self, fpath, dataindex):
        """
        :param fpath: The path to the file to read from.
        :param dataindex: A parameter that indicates which index in the tuple of data items on a line to get.
        """
        self.fpath = fpath
        self.dataindex = dataindex
        self.epoch_num = 0

    def get(self):
        epoch_str = lambda epnum : "----- " + str(epnum) + " -----"
        with open(self.fpath) as f:
            # Take only the lines after the most recent epoch
            lines = [line.strip() for line in itertools.dropwhile(lambda x: x.strip() != epoch_str(self.epoch_num), f)]
            lines = [line.strip() for line in itertools.takewhile(lambda x: x.strip() != epoch_str(self.epoch_num + 1), lines)]
            lines = [line for line in lines if line.strip() != "" and not line.startswith('-')]
            tups = [line.split(',') for line in lines]
            data = [float(tup[self.dataindex].strip()) for tup in tups]
        if data:
            self.epoch_num += 1
        return data

if __name__ == "__main__":
    loss = 0
    acc = 1
    g = Getter(FILE_PATH, loss)
    plotter = Plotter([], [], g)
    plotter.animate()

