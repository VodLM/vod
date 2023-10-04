# Install nvidia drivers (GCP)
# sudo /opt/deeplearning/install-driver.sh

# Check if CentOS(yum) or Debian(apt) based
if [ -f /etc/debian_version ]; then
    echo "Debian based"
    PKGMD="apt-get"
    sudo $PKGMD update
    sudo $PKGMD install -y zsh git htop util-linux gcc make libaio-dev
elif [ -f /etc/redhat-release ]; then
    echo "CentOS based"
    PKGMD="yum"
    sudo $PKGMD update
    sudo $PKGMD install -y zsh git htop util-linux-user gcc kernel-devel libblas-dev
else
    echo "Unknown OS, defaulting to apt-get"
    PKGMD="apt-get"
fi

# Test
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX # check for `GLIBCXX_3.4.30`

# Oh my zsh
sudo chsh -s $(which zsh) $(whoami)
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Enable scrolling
echo "termcapinfo xterm* ti@:te@" >>~/.screenrc


# Install CUDA | find your version at `https://developer.nvidia.com/cuda-downloads`
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run

# Install mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
mamba init

# Setup mamba env
mamba create --name vod python=3.10
echo 'mamba activate vod' >>~/.zshrc
source ~/.zshrc
mamba install -y -c nvidia cuda
mamba install -y -c conda-forge cudnn cudatoolkit
pip install pip poetry gpustat nvitop

# Install and setup elasticsearch
if [[ $PKGMD != "apt-get" ]]; then
    wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.7.1-x86_64.rpm
    wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.7.1-x86_64.rpm.sha512
    shasum -a 512 -c elasticsearch-8.7.1-x86_64.rpm.sha512
    sudo rpm --install elasticsearch-8.7.1-x86_64.rpm
else
    wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo gpg --dearmor -o /usr/share/keyrings/elasticsearch-keyring.gpg
    sudo apt-get install apt-transport-https
    echo "deb [signed-by=/usr/share/keyrings/elasticsearch-keyring.gpg] https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee /etc/apt/sources.list.d/elastic-8.x.list
    sudo apt-get update
    sudo apt-get install elasticsearch

# Instlall ICU plugins
sudo bin/elasticsearch-plugin install analysis-icu

# (Optional) Setup elasticsearch config: open file and comment out the security stuff
sudo nano /etc/elasticsearch/elasticsearch.yml
# Setup elasticsearch jvm: limit the virtual memory to 320GB
es_opts='
-Xms32g
-Xmx64g
'
sudo echo "$es_opts" | sudo tee /etc/elasticsearch/jvm.options.d/cfg.options

# Start ES as a service
sudo systemctl daemon-reload
sudo systemctl enable elasticsearch.service
sudo systemctl stop elasticsearch.service
sudo systemctl start elasticsearch.service


# Environment
nano ~/.raffle/gc-mlm-100322.json # <- Put your Google credentials here
nano ~/.raffle/.env # <- Copy the rest of your env variables here >

# Auto-load env variables
line='export $(grep -v "^#" ~/.raffle/.env | xargs)'
echo $line >>~/.zshrc
source ~/.zshrc


# Start Qdrant (Docker)
docker run -p 6333:6333 -p 6334:6334 \
    --detach \
    --restart always \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant:latest