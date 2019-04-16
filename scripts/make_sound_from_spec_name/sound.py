"""
Takes the filename of a preprocessed spectrogram and creates the sound file
that created it.
"""
import audiosegment as asg
import os
import sys

def segment_from_specname(specfname: str, long: bool) -> asg.AudioSegment:
    """
    Returns an AudioSegment object that corresponds to the given spectrogram filename.

    If `long` is True, we use 16 kHz sample rate and 0.5 second long slices.
    Otherwise, we use 8 kHz sample rate and 0.3 second long slices.
    """
    specfname = os.path.basename(specfname)         # english_10240.wav_21.png

    wavindex_plus_png = specfname.split('_')[-1]    # 21.png
    wavindex = int(wavindex_plus_png.split('.')[0]) # 21

    wavfname = specfname.split('.')[0] + ".wav"     # english_10240.wav
    wavfpath = "/media/max/seagate8TB/thesis_audio/preprocessed_gold_data/" + wavfname

    if not os.path.isfile(wavfpath):
        raise ValueError("Cannot find file: {}".format(wavfpath))

    if long:
        sample_rate_hz = 16000.0    # 16 kHz sample rate
        ln = 0.5                    # seconds
    else:
        sample_rate_hz = 8000.0     # 8 kHz sample rate
        ln = 0.3                    # seconds

    bytewidth       = 2          # 16-bit samples
    nchannels       = 1          # mono
    seg = asg.from_file(wavfpath)
    segments = seg.dice(ln)
    for i, s in enumerate(segments):
        if i == wavindex:
            return s.resample(sample_rate_hz, bytewidth, nchannels)
    raise ValueError("After dicing by {} seconds, can't find index segment indexed {}".format(ln, wavindex))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python {} <path-to-preprocessed-spectrogram> <long or short>".format(sys.argv[0]))
        exit(1)
    elif sys.argv[2].strip().lower() not in ("long", "short"):
        print("Second argument must be either 'long' or 'short' (for the length of the spectrogram)")
        exit(3)

    seg = segment_from_specname(sys.argv[1], sys.argv[2].strip().lower() == "long")
    seg.export(os.path.basename(sys.argv[1]) + ".wav", format="WAV")
