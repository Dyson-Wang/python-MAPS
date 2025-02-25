#!/bin/bash

sudo apt-get update
# sudo apt-get upgrade -qq
sudo apt-get -y install apt-transport-https ca-certificates curl criu software-properties-common python3-pip virtualenv python3-setuptools sysbench ioping linux-tools-generic linux-tools-$(uname -r) linux-cloud-tools-$(uname -r) apt-transport-https ca-certificates curl gnupg-agent software-properties-common
# sudo apt-get -y criu
# sudo apt-get -y install linux-tools-generic linux-tools-$(uname -r) linux-cloud-tools-$(uname -r)
# sh ./Anaconda3-2024.06-1-Linux-x86_64.sh
python3 -m pip install flask-restful inotify Flask psutil docker

sudo chmod +x $HOME/agent/agent.py

# Install Docker

sudo mkdir /etc/docker
sudo cp $HOME/agent/scripts/daemon.json /etc/docker
# sudo apt-get -y install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
sudo install -m 0755 -d /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o ~/docker.asc
sudo mv ~/docker.asc /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
# sudo apt-get update
# sudo apt-get -y install docker-ce docker-ce-cli containerd.io
# sudo groupadd docker 
sudo usermod -aG docker $USER

# Setup Flask 

sudo cp ~/agent/scripts/flask.conf /etc/init.d/
sudo cp ~/agent/scripts/flask.service /lib/systemd/system/flask.service
sudo chmod +x ~/agent/scripts/delete.sh

# Load Docker images

sudo systemctl daemon-reload
sudo systemctl restart docker
# sudo docker info

cd ~/agent/docimgs

sudo docker load -i shreshthtuli.tar

# sudo rm -f shreshthtuli.tar
# sudo rm -f /swapfile

sudo mkdir ~/container_data/
sudo chown $USER ~/container_data/

# install criu
echo "deb http://archive.ubuntu.com/ubuntu noble main universe
deb http://archive.ubuntu.com/ubuntu noble-updates main universe
deb http://archive.ubuntu.com/ubuntu noble-security main universe" | sudo tee -a /etc/apt/sources.list
sudo apt update
# sudo apt install criu -qq
sudo apt install -y python3-flask-restful python3-inotify python3-flask python3-psutil python3-docker

sudo service flask start
sudo systemctl enable flask