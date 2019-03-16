model=$1
datadir="/media/max/seagate8TB/thesis_audio/test_spectrogram_images/test_set/useless_subdirectory"
python3 testvae.py $model $datadir/english_16641.wav_32.png
python3 plotvae.py $model $datadir
