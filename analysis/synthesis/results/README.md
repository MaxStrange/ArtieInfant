# Results

This folder contains directories of genetic algorithm results for the articulatory synthesizer.

Each directory may contain:

- A .png showing the wave forms of the evolution of the sound. The bottom wave form in the image is the target.
- A target sound in OGG or WAV format.
- Phase0OutputSound.wav: The sound file output after pretraining is completed
- Phase0OutputSound.csv: The history file for Phase 0. This can be graphed to show how min, max, and avg agents in the
  gene pool changed over time.
- Phase1Output_X.wav: The sound file output after the X'th articulator group. At the time of this writing, the articulator
  group order was: jaw, nasal, lingual-support, lingual-tongue, labial.
- Phase1Output_X.csv: Same as Phase0OutputSound.csv, but for the X'th training epoch in Phase 1.
