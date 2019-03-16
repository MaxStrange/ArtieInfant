python ae.py /home/maxst/repos/ArtieInfant/scripts/test_vae/example_spectrograms -v -b 2 -e $1
python testvae.py /home/maxst/repos/ArtieInfant/scripts/autoencoder/models/VAE.h5 /home/maxst/repos/ArtieInfant/scripts/test_vae/example_spectrograms/english_3481.wav_17.png
python plotvae.py /home/maxst/repos/ArtieInfant/scripts/autoencoder/models/VAE.h5 /home/maxst/repos/ArtieInfant/scripts/test_vae/example_spectrograms
