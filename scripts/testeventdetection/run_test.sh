if [ $# -ne 1 ]; then
    echo USAGE: $0 path/to/test/model
    exit 1
fi

PYTHONPATH=$PYTHONPATH:/home/max/git_repos/ArtieInfant/scratch/mlpvad/src/models python3 ./eventdetectiontest.py $1 debug.wav
