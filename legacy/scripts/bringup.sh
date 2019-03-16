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
    python3-dev \
    python3-pip \
    ssh \
    -y

git config --global core.editor vim
git config --global push.default simple
git config --global user.name MaxStrange

sudo systemctl start ssh

if [ ! -d "/home/max/repos" ]; then
    mkdir /home/max/repos
fi

cd /home/max/repos

echo "Installing your vimrc"
git clone https://www.github.com/MaxStrange/myvim.git
cd myvim
./install.sh
cp .vimrc /home/max/
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
echo 'PATH=$HOME/.local/bin:$PATH' >> /home/max/.bashrc
echo 'export WORKON_HOME=$HOME/.virtualenvs' >> /home/max/.bashrc
echo 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' >> /home/max/.bashrc
echo 'source $HOME/.local/bin/virtualenvwrapper.sh' >> /home/max/.bashrc

echo "Don't forget to reserve this device's IP address in the router."
echo "You should now source your .bashrc, then mkvirtualenv ai"

