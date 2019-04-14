# -------- Short time RMS genetic experiments --------
#echo "!!!!!!!!!!!!!! genetic2/1.cfg !!!!!!!!!!!!!!!!!!!!"
#python3 main.py experiment/configfiles/genetic2/1.cfg --pretrain-synth --loglevel info
#echo "!!!!!!!!!!!!!! genetic2/2.cfg !!!!!!!!!!!!!!!!!!!!"
#python3 main.py experiment/configfiles/genetic2/2.cfg --pretrain-synth --loglevel info
#echo "!!!!!!!!!!!!!! genetic2/3.cfg !!!!!!!!!!!!!!!!!!!!"
#python3 main.py experiment/configfiles/genetic2/3.cfg --pretrain-synth --loglevel info
#echo "!!!!!!!!!!!!!! genetic2/4.cfg !!!!!!!!!!!!!!!!!!!!"
#python3 main.py experiment/configfiles/genetic2/4.cfg --pretrain-synth --loglevel info

# -------- Short time XCOR genetic experiments ------
echo "!!!!!!!!!!!!!! genetic2/5.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic2/5.cfg --pretrain-synth --train-synth --loglevel info
echo "!!!!!!!!!!!!!! genetic2/6.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic2/6.cfg --pretrain-synth --train-synth --loglevel info
echo "!!!!!!!!!!!!!! genetic2/7.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic2/7.cfg --pretrain-synth --train-synth --loglevel info
echo "!!!!!!!!!!!!!! genetic2/8.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic2/8.cfg --pretrain-synth --train-synth --loglevel info

# --------- Long time baseline genetic experiments --
echo "!!!!!!!!!!!!!! genetic3/1.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic3/1.cfg --pretrain-synth --train-synth --loglevel info
echo "!!!!!!!!!!!!!! genetic3/2.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic3/2.cfg --pretrain-synth --train-synth --loglevel info
echo "!!!!!!!!!!!!!! genetic3/3.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic3/3.cfg --pretrain-synth --train-synth --loglevel info
echo "!!!!!!!!!!!!!! genetic3/4.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic3/4.cfg --pretrain-synth --train-synth --loglevel info

# --------- Short time baseline genetic experiments -
echo "!!!!!!!!!!!!!! genetic3/5.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic3/5.cfg --pretrain-synth --train-synth --loglevel info
echo "!!!!!!!!!!!!!! genetic3/6.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic3/6.cfg --pretrain-synth --train-synth --loglevel info
echo "!!!!!!!!!!!!!! genetic3/7.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic3/7.cfg --pretrain-synth --train-synth --loglevel info
echo "!!!!!!!!!!!!!! genetic3/8.cfg !!!!!!!!!!!!!!!!!!!!"
python3 main.py experiment/configfiles/genetic3/8.cfg --pretrain-synth --train-synth --loglevel info

# --------- Short time vanilla latent dims rerun ----
echo "!!!!!!!!!!!!!! latentdims/13.cfg !!!!!!!!!!!!!!!!!!!!"
python main.py experiment/configfiles/latentdims/13.cfg --train-vae --loglevel info
echo "!!!!!!!!!!!!!! latentdims/14.cfg !!!!!!!!!!!!!!!!!!!!"
python main.py experiment/configfiles/latentdims/14.cfg --train-vae --loglevel info
echo "!!!!!!!!!!!!!! latentdims/15.cfg !!!!!!!!!!!!!!!!!!!!"
python main.py experiment/configfiles/latentdims/15.cfg --train-vae --loglevel info
echo "!!!!!!!!!!!!!! latentdims/16.cfg !!!!!!!!!!!!!!!!!!!!"
python main.py experiment/configfiles/latentdims/16.cfg --train-vae --loglevel info

# --------- Short time Euclid genetic experiments ---
echo "!!!!!!!!!!!!!! closedloop/3.cfg !!!!!!!!!!!!!!!!!!!!"
python main.py experiment/configfiles/closedloop/3.cfg --pretrain-synth --train-synth --loglevel info
echo "!!!!!!!!!!!!!! closedloop/4.cfg !!!!!!!!!!!!!!!!!!!!"
python main.py experiment/configfiles/closedloop/4.cfg --pretrain-synth --train-synth --loglevel info

# --------- Long time Overfitting -------------------

# --------- Short time Underfitting -----------------

# --------- Variational Loss Function ---------------
