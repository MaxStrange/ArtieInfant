"""
Script for testing the efficacy of given models at audio event detection.
"""
import audiosegment
import keras
import numpy as np

class Model:
    def __init__(self, modelpath):
        self.model = keras.models.load_model(modelpath)

    def predict(self, seg):
        raise NotImplementedError("This function needs to be overridden in a subclass")

class FFTModel(Model):
    def __init__(self, modelpath, normalize):
        self.normalize = normalize

    def predict(self, seg):
        """
        Returns a 0 if the event is not detected and 1 if the event is detected in the given segment.
        """
        _hist_bins, hist_vals = seg.fft()
        real_normed = np.abs(hist_vals) / len(hist_vals)
        if self.normalize:
            real_normed = (real_normed - min(real_normed)) / (max(real_normed) + 1E-9)
        np.expand_dims(real_normed, axis=0)  # Add batch dimension
        prediction = self.model.predict(real_normed)
        return prediction
 

if __name__ == "__main__":
    # TODO
    # load the model
    # load some golden test data
    # apply the model to the data
    # save the results for human checking
    pass