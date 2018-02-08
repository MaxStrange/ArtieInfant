#! /bin/sh
# This script is for bringing up a new node in the Kafka cluster.
# This should be all you need to run in order for a node to be ready
# to go.
if [ "$EUID" -ne 0 ]
    then echo "Please run as root."
    exit 1
fi

apt-get update
apt-get install git curl vim -y
git config --global core.editor vim
git config --global push.default simple
git config --global user.name MaxStrange

if [ ! -d "~/repos" ]; then
    mkdir ~/repos
fi

cd ~/repos

echo "Installing your vimrc"
git clone https://www.github.com/MaxStrange/myvim.git
cd myvim
./install.sh
cp .vimrc ~/
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
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3" >> ~/.bashrc
echo "source ~/.local/bin/virtualenvwrapper.sh" >> ~/.bashrc

