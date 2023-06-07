# install nvidia drivers (GCP)
# sudo /opt/deeplearning/install-driver.sh

# check if CentOS(yum) or Debian(apt) based
if [ -f /etc/debian_version ]; then
    echo "Debian based"
    PKGMD="apt-get"
    sudo $PKGMD update
    sudo $PKGMD install -y zsh git htop util-linux gcc make
elif [ -f /etc/redhat-release ]; then
    echo "CentOS based"
    PKGMD="yum"
    sudo $PKGMD update
    sudo $PKGMD install -y zsh git htop util-linux-user gcc kernel-devel
else
    echo "Unknown OS, defaulting to apt-get"
    PKGMD="apt-get"
fi


# oh my zsh
sudo chsh -s $(which zsh) $(whoami)
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# enable scrolling
echo "termcapinfo xterm* ti@:te@" >>~/.screenrc


# install CUDA | find your version at `https://developer.nvidia.com/cuda-downloads`
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sudo sh cuda_12.1.1_530.30.02_linux.run

# install mamba
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
mamba init

# setup mamba env
mamba create --name dev python=3.10
echo 'mamba activate dev' >>~/.zshrc
source ~/.zshrc
mamba install -y -c nvidia cuda
mamba install -y -c conda-forge cudnn cudatoolkit==12.1
pip install gpustat nvitop

# install and setup elasticsearch
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

# setup elasticsearch config: open file and comment out the security stuff
sudo nano /etc/elasticsearch/elasticsearch.yml
# setup elasticsearch jvm: limit the virtual memory to 64GB
es_opts='
-Xms64g
-Xmx320g
'
sudo echo "$es_opts" | sudo tee /etc/elasticsearch/jvm.options.d/cfg.options

# start ES
sudo systemctl daemon-reload
sudo systemctl enable elasticsearch.service
sudo systemctl stop elasticsearch.service
sudo systemctl start elasticsearch.service
# sudo systemctl stop elasticsearch.service


# install poetry
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="/home/vlievin/.local/bin:$PATH"' >>~/.zshrc
source ~/.zshrc


gcp='...'
mkdir ~/.raffle/
echo "$gcp" >~/.raffle/gc-mlm-100322.json


ENV="
NAMESPACE=local
# < copy the rest of your env variables here >s
"
echo "$ENV" >~/.raffle/.env
line='export $(grep -v "^#" ~/.raffle/.env | xargs)'
echo $line >>~/.zshrc
source ~/.zshrc