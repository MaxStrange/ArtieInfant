# Notes

1. Try preprocessing all the data and dumping it into one big file. See how big that is. Maybe it is big enough to fit into memory.
1. Check memory leakage while dumping the preprocessed data. If it isn't there, then it's a problem with interaction with Keras (or Keras itself).
1. Seem to be spawning lots of processes - figure out why and if it is related.
1. Play with the spectrogram parameters until you can actually see a phoneme spectrogram, then get it as low-dimensioned as you can from there while
    still being able to tell.
1. Read Chapters from the DSP text book (6.5 and 6.6)