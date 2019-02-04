# Dataprovider

This folder contains all the code required to yield up the data to the rest of
the scripts and modules, as well as to the trained module during inference time.

## Note

There is a pretty serious memory leak somewhere in here - featureprovider or dataprovider most likely.
I haven't had time to look into it much, but it seems like one or both of these is spawning way
too many Python processes, which is probably related.