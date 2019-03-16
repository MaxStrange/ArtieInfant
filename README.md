# ArtieInfant

[![Build Status](https://travis-ci.org/MaxStrange/ArtieInfant.svg?branch=master)](https://travis-ci.org/MaxStrange/ArtieInfant)

Simulated Infant Babbling

**There will be a lot more explanation once I am done writing my thesis**

## What is this?

This is one of the deliverables for my Master's thesis at the University of Washington. It is a computational model that seeks to test
a few key parts of a theory I developed about how infants acquire the sound system of their native language(s).

The general usage is this:

1. Get a WHOLE BUNCH of raw audio data (ideally audio that an infant is likely to hear in the first eight months or so of life).
1. Write a config file. Or, more reasonably, adapt your own from the existing thesis.cfg in ArtieInfant/Artie/experiment/configfiles/.
1. `cd ArtieInfant/Artie`
1. `python3 main.py <args - pass in the config file>`
1. Analyze the results with scripts in the ArtieInfant/analysis folder.

The general idea is that this simulates an infant learning the sound system of their native language over the first six to eight months.
The infant will learn to distinguish between different sounds, and will learn to produce some on its own.

## Where do I get the data?

The data that I have used is my own. I recorded the auditory environment for several hours a day around my son from the first week of his
life to when he was about eight months old... because I'm crazy... Unfortunately, I can't open source this mountain of data, since I would
first need to go through it and make sure everyone who was ever recorded is either redacted or only says really uninteresting things.
Frankly, this would be quite an undertaking, so it isn't going to happen.

This means that you will have to get your own data if you (for whatever reason) want to run this model yourself. See the config file
for what I preprocessed the data to (in terms of sampling rate, etc.).

## What do I do when I have data?

Preprocess it! It needs to be converted into spectrograms. Take a look at the config files to see what I did.

## What is in store for the future?

With this particular repository, not whole lot. I need to actually finish this work (I graduate in May of 2019), but
after that, I am thinking I would like to take this work and re-implement it in an embedded context, so that I can
actually embody ArtieInfant. We'll see.

## License

The license is MIT, so go ahead and do whatever you want! Just make sure you give me credit (mostly because I don't want you to be dishonest, not because I care about the credit :P).
