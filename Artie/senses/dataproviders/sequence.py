"""
Module for exposing Sequence implementation.
"""
import keras

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class Sequence(keras.utils.Sequence):
    """
    Implementation for keras.utils.Sequence.
    """
    def __init__(self, ms_of_dataset, ms_per_batch, nworkers, provider_fun, *args, **kwargs):
        self.ms_of_dataset = ms_of_dataset
        self.ms_per_batch  = ms_per_batch
        self.provider      = provider_fun(*args, **kwargs)
        self.nworkers      = nworkers

    def __len__(self):
        return self.ms_of_dataset / self.ms_per_batch

    def __next__(self):
        # Return a batch from the batch producer
        return next(self.provider)