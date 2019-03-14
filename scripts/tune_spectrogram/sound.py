"""
Takes the filename of a preprocessed spectrogram and creates the sound file
that created it.
"""
import audiosegment as asg
import os
import sys

def segment_from_specname(specfname: str) -> asg.AudioSegment:
    """
    Returns an AudioSegment object that corresponds to the given spectrogram filename.
    """
    specfname = os.path.basename(specfname)         # english_10240.wav_21.png

    wavindex_plus_png = specfname.split('_')[-1]    # 21.png
    wavindex = int(wavindex_plus_png.split('.')[0]) # 21

    wavfname = specfname.split('.')[0] + ".wav"     # english_10240.wav
    wavfpath = "/media/max/seagate8TB/thesis_audio/preprocessed_gold_data/" + wavfname

    if not os.path.isfile(wavfpath):
        raise ValueError("Cannot find file: {}".format(wavfpath))

    sample_rate_hz  = 16000.0    # 16kHz sample rate
    bytewidth       = 2          # 16-bit samples
    nchannels       = 1          # mono
    seg = asg.from_file(wavfpath)
    segments = seg.dice(0.5)
    for i, s in enumerate(segments):
        if i == wavindex:
            return s.resample(sample_rate_hz, bytewidth, nchannels)
    raise ValueError("After dicing by 0.5 seconds, can't find index segment indexed {}".format(wavindex))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: python {} <path-to-preprocessed-spectrogram>".format(sys.argv[0]))
        exit(1)
    elif not os.path.isfile(sys.argv[1]):
        print("USAGE: python {} <path-to-preprocessed-spectrogram>".format(sys.argv[0]))
        exit(2)

    seg = segment_from_specname(sys.argv[1])
    seg.export(os.path.basename(sys.argv[1]) + ".wav", format="WAV")
