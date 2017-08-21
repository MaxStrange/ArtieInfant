"""
Trim each wav file to one minute.
"""
import os
import sys

MS_PER_S = 1000
S_PER_MIN = 60
MS_PER_MIN = MS_PER_S * S_PER_MIN

def _minutetrim_single(wav_in):
    starts = range(0, int(round(wav_in.duration_seconds * MS_PER_S)), MS_PER_MIN)
    stops = (min(wav_in.duration_seconds * MS_PER_S, start + MS_PER_MIN) for start in starts)
    wav_outs = [wav_in[start:stop] for start, stop in zip(starts, stops)]

    # Now cut out the last three seconds of the last item in wav_outs (it will just be microphone artifact)
    # or, if the last item is less than three seconds, just get rid of it
    if wav_outs[-1].duration_seconds > 3:
        wav_outs[-1] = wav_outs[-1][:-MS_PER_S * 3]
    else:
        wav_outs = wav_outs[:-1]

    return wav_outs

def minutetrim(wav_ins):
    """
    Takes each wav file and splits it up into a list of one-minute (or less) wav files.

    Returns:
        [[wav_file, wav_file], [wav_file], etc.]
    """
    wav_outs = [_minutetrim_single(w_in) for w_in in wav_ins]
    wav_outs_segments = []
    for minute_trims in wav_outs:
        for minute_wav in minute_trims:
            wav_outs_segments.append(minute_wav)
    return wav_outs_segments

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3", sys.argv[0], os.sep.join(["path", "to", "file.wav"]))
        exit(-1)
    elif not os.path.isfile(sys.argv[1]):
        print(str(sys.argv[1]), "is not a valid path to a file.")
        exit(-1)

    fpath = sys.argv[1]
    trimmed = minutetrim([fpath])
    print(trimmed)

