# This script is for bringing up a new node in the Kafka cluster.
# This should be all you need to run in order for a node to be ready
# to go.
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install \
    git \
    curl \
    vim \
    sox \
    ffmpeg \
    python3-tk \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    -y

git config --global core.editor vim
git config --global push.default simple
git config --global user.name MaxStrange

if [ ! -d "/home/pi/repos" ]; then
    mkdir /home/pi/repos
fi

cd /home/pi/repos

echo "Installing your vimrc"
git clone https://www.github.com/MaxStrange/myvim.git
cd myvim
./install.sh
cp .vimrc /home/pi/
cd ..

echo "Installing ArtieInfant"
git clone https://www.github.com/MaxStrange/ArtieInfant.git
cd ArtieInfant

echo "Downloading Kafka"
wget http://apache.claz.org/kafka/1.0.0/kafka_2.12-1.0.0.tgz
tar -zxvf kafka_2.12-1.0.0.tgz
rm kafka_2.12-1.0.0.tgz

echo "Installing virtualenvwrapper"
pip3 install --user virtualenv virtualenvwrapper
echo 'PATH=/home/pi/.local/bin:$PATH' >> /home/pi/.bashrc
echo 'export WORKON_HOME=$HOME/.virtualenvs' >> /home/pi/.bashrc
echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' >> /home/pi/.bashrc
echo 'source /home/pi/.local/bin/virtualenvwrapper.sh' >> /home/pi/.bashrc

echo "You should now source your .bashrc, then mkvirtualenv ai"

