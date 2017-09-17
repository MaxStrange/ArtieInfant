import audiosegment
import keras
import keras.models
import matplotlib.pyplot as plt
import numpy as np
import os
try:
    import metrics
except ImportError:
    print("You must provide a PYTHONPATH to the metrics module in the source directory.")
    exit(1)
import sys

class Model:
    def __init__(self, modelpath):
        self.model = keras.models.load_model(modelpath, custom_objects={"fscore": metrics.fscore})

    def predict(self, seg):
        _bins, fft_vals = seg.fft()
        fft_vals = np.abs(fft_vals) / len(fft_vals)
        fft_vals = (fft_vals - min(fft_vals)) / (max(fft_vals) + 1E-9)
        predicted_np_form = self.model.predict(np.array([fft_vals]), batch_size=1)
        prediction_as_int = int(round(predicted_np_form[0][0]))
        return prediction_as_int

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: python3", sys.argv[0], os.sep.join("path to model".split(' ')),
              os.sep.join("path to wav file".split(' ')))
        exit(1)

    modelpath = sys.argv[1]
    wavpath = sys.argv[2]
    assert os.path.isfile(modelpath)
    assert os.path.isfile(wavpath)

    print("Loading model...")
    model = Model(modelpath)
    print("Loading audio file...")
    seg = audiosegment.from_file(wavpath).resample(sample_rate_Hz=32000, sample_width=2, channels=1)
    print("Running test...")
    pyes_to_no = 0.3
    pno_to_yes = 0.2
    ptrue_pos_rate=0.8
    pfalse_neg_rate=0.2
    events = seg.detect_event(model, ms_per_input=30, transition_matrix=[pyes_to_no, pno_to_yes],
                     model_stats=[ptrue_pos_rate, pfalse_neg_rate], event_length_s=0.25, prob_raw_yes=0.7)
    nos = [event[1] for event in events if event[0] == 'n']
    yeses = [event[1] for event in events if event[0] == 'y']
    if len(nos) > 1:
        print("Saving the audio where the event was not detected into notdetected.wav")
        notdetected = nos[0].reduce(nos[1:])
        notdetected.export("notdetected.wav", format="WAV")
        plt.subplot(211)
        plt.title("Not Detected")
        plt.plot(notdetected.get_array_of_samples())
    if len(yeses) > 1:
        print("Saving the audio where the event WAS detected into detected.wav")
        detected = yeses[0].reduce(yeses[1:])
        detected.export("detected.wav", format="WAV")
        plt.subplot(212)
        plt.title("Detected")
        plt.plot(detected.get_array_of_samples())
    if len(nos) > 1 or len(yeses) > 1:
        plt.tight_layout()
        plt.show()
