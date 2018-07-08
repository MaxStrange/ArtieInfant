"""
Script for testing the efficacy of given models at audio event detection.
"""
import argparse
import audiosegment
import keras
import numpy as np
import os

class Model:
    def __init__(self, modelpath):
        self.model = keras.models.load_model(modelpath)

    def predict(self, seg):
        raise NotImplementedError("This function needs to be overridden in a subclass")

class FFTModel(Model):
    def __init__(self, modelpath, normalize):
        super().__init__(modelpath)
        self.normalize = normalize

    def predict(self, seg):
        """
        Returns a 0 if the event is not detected and 1 if the event is detected in the given segment.
        """
        _hist_bins, hist_vals = seg.fft()
        real_normed = np.abs(hist_vals) / len(hist_vals)
        if self.normalize:
            real_normed = (real_normed - min(real_normed)) / (max(real_normed) + 1E-9)
        prediction = self.model.predict(np.array([real_normed]), batch_size=1)
        prediction_as_int = int(round(prediction[0][0]))
        return prediction_as_int

class SprectrogramModel(Model):
    def __init__(self, modelpath, window_length_ms, overlap, normalize):
        super().__init__(modelpath)
        self.window_length_ms = window_length_ms
        self.overlap = overlap
        self.normalize = normalize

    def predict(self, seg):
        """
        Returns a 0 if the event is not detected and 1 if the event is detected in the given segment.
        """
        _hist_bins, _times, amplitudes = seg.spectrogram(window_length_s=self.window_length_ms/1000, overlap=self.overlap)
        amplitudes_real_normed = np.abs(amplitudes) / len(amplitudes)
        if self.normalize:
            amplitudes_real_normed = np.apply_along_axis(lambda v: (v - min(v)) / (max(v) + 1E-9), 1, amplitudes_real_normed)

        amplitudes_real_normed = np.expand_dims(amplitudes_real_normed, axis=-1)  # Add batch dimension
        prediction = self.model.predict(np.array([amplitudes_real_normed]))
        prediction_as_int = int(round(prediction[0][0]))
        return prediction_as_int
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modelpath", type=str, help="Path to model to use.")
    parser.add_argument("mode", type=str, choices=("baby", "language", "voice", "test"), help="Model to train.")
    parser.add_argument("-fp", "--file_path", type=str, default=None, help="The path to the file to detect the event in. If unspecified, tries to find some in a default location.")
    parser.add_argument("-pf", "--provider_fun", type=str, choices=("fft", "spectrogram", "sequence"), default="fft", help="The type of model to train.")
    parser.add_argument("-el", "--event_length_ms", type=float, default=500, help="The typical length of the event in ms.")
    parser.add_argument("-pv", "--positive_predictive_value", type=float, default=0.5, help="The model's positive predictive value.")
    parser.add_argument("-nv", "--negative_predictive_value", type=float, default=0.5, help="The model's negative predictive value.")
    parser.add_argument("-sr", "--sample_rate", type=int, default=24E3, help="Sample rate in Hz to resample the data to.")
    parser.add_argument("-nc", "--nchannels", type=int, default=1, help="Number of audio channels to resample to.")
    parser.add_argument("-bw", "--bytewidth", type=int, default=2, help="Number of bytes per sample to resample the audio to.")
    parser.add_argument("-ms", "--ms_per_model_input", type=float, default=100.0, help="Number of ms of audio to use as the input to the model.")
    parser.add_argument("-nz", "--normalize", type=bool, default=True, help="Only valid when provider_fun is 'fft' or 'spectrogram'. Normalize each FFT or spectrogram.")
    parser.add_argument("-so", "--spec_overlap", type=float, default=0.2, help="Only valid when provider_fun is 'spectrogram'. The fraction of overlap of each spectrogram window.")
    parser.add_argument("-sw", "--ms_per_spec_window", type=float, default=None, help="Only valid when provider_fun is 'spectrogram'. Length of each FFT in the time domain used to create the spectrogram. Defaults to 1/100th of the overall window.")
    args = parser.parse_args()

    # Check if modelpath is valid
    modelpath = os.path.abspath(args.modelpath)
    if not modelpath:
        print("{} is not a file. Need a path to a model file.".format(modelpath))
        exit(1)

    # load the model
    if args.provider_fun == "fft":
        model = FFTModel(modelpath, args.normalize)
    elif args.provider_fun == "spectrogram":
        model = SprectrogramModel(modelpath, args.ms_per_spec_window, args.spec_overlap, args.normalize)
    elif args.provider_fun == "sequence":
        raise NotImplementedError("No Sequence model implementation yet.")
    else:
        raise ValueError("{} is not an allowed argument. Argparse should have caught this...".format(args.provider_fun))

    # load some golden test data
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "test_data_directory")
    golden_reference_data = {
        "baby": "/mnt/data/thesis_audio/baby_detection/test/BABY_more_cooing/Baby_cooing-QOaqcjY93RA_seg0.wav",
        "language": "/mnt/data/thesis_audio/engchin/test/ZH_chinese_talk_show_kaze_ninja/Thj_Johnny_-_bTLgiEH-omY47v62oJs_seg0.wav",
        "voice": "/mnt/data/thesis_audio/voice_detection/test/VOICE_uk_problems_arguments_etc/Argue_with_Old_man_on_London_Bus-jJeBa2ZzTy8_seg0.wav",
        "test": os.path.join(test_dir, "babies", "baby_laughter.wav"),
    }
    if args.file_path is None:
        reference_data = golden_reference_data[args.mode]
    elif os.path.isfile(args.file_path):
        reference_data = args.file_path
    else:
        print("Given file path {}, but does not point to a real file.".format(args.file_path))
        exit(1)
    segment = audiosegment.from_file(reference_data)
    segment = segment.resample(sample_rate_Hz=args.sample_rate, sample_width=args.bytewidth, channels=args.nchannels)

    # apply the model to the data
    p_yes_to_no = 0.6
    p_no_to_yes = 0.4
    matrix = [p_yes_to_no, p_no_to_yes]
    ppv = args.positive_predictive_value
    npv = args.negative_predictive_value
    model_stats = [ppv, npv]
    event_length_s = args.event_length_ms / 1000
    events = segment.detect_event(model, args.ms_per_model_input, matrix, model_stats, event_length_s)
    positives = [tup[1] for tup in events if tup[0] == 'y']
    negatives = [tup[1] for tup in events if tup[0] == 'n']

    if len(positives) > 1:
        positives = positives[0].reduce(positives[1:])
    elif len(positives) == 1:
        positives = positives[0]

    if len(negatives) > 1:
        negatives = negatives[0].reduce(negatives[1:])
    elif len(negatives) == 1:
        negatives = negatives[0]

    # save the results for human checking
    if positives:
        positives.export("detection_test_{}_{}_positives.wav".format(args.mode, args.provider_fun), format="wav")
    if negatives:
        negatives.export("detection_test_{}_{}_negatives.wav".format(args.mode, args.provider_fun), format="wav")
