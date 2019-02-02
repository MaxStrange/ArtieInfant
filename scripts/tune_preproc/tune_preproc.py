"""
This script is useful for tuning the preprocessing steps.
"""
import audiosegment as asg
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Need path to audio file")
        exit(1)
    elif not os.path.isfile(sys.argv[1]):
        print("Need path to audio file")
        exit(2)

    print("Loading in the audiofile...")
    seg = asg.from_file(sys.argv[1])
    print("Done")

    #############################################################
    ## Parameters for Tuning ##
    silence_duration_s        = 5.0
    silence_threshold_percent = 5.0
    #############################################################

    # Remove the silence
    seg = seg.filter_silence(duration_s=silence_duration_s, threshold_percentage=silence_threshold_percent)

    # Save
    seg.export("output.wav", format="WAV")
