#!/bin/bash
# scp -r suyo@172.21.15.56:/home/suyo/wspace/code/sp/workflow/agent .
echo "Acquire::http::Proxy \"http://172.21.15.146:2080/\";
Acquire::https::Proxy \"http://172.21.15.146:2080/\";" | sudo tee /etc/apt/apt.conf.d/proxy.conf

cat $HOME/agent/id_rsa.pub >> ~/.ssh/authorized_keys
sudo chmod 600 $HOME/agent/id_rsa
sudo chmod +x $HOME/agent/scripts/calIPS.sh
sudo chmod +x $HOME/agent/scripts/calIPS_clock.sh
sudo chmod +x $HOME/agent/scripts/setup.sh
sed -i -e 's/\r$//' $HOME/agent/scripts/setup.sh
echo "export proxy=\"http://172.21.15.146:2080\"
export http_proxy=\"http://172.21.15.146:2080\"
export https_proxy=\"http://172.21.15.146:2080\"
export ftp_proxy=\"http://172.21.15.146:2080\"
export no_proxy=\"localhost, 127.0.0.1, ::1, 192.168.0.0/16, 172.16.0.0/12, 10.0.0.0/8\"" >> ~/.bashrc
sudo apt install vim net-tools curl wget openssh-server -y -qq
sh ~/agent/scripts/Anaconda3-2024.06-1-Linux-x86_64.sh
# sudo vim /etc/needrestart/needrestart.conf # $nrconf{restart} = 'a';
# sudo EDITOR=vim visudo # vagrant ALL=(ALL) NOPASSWD: ALL 其他的也写成NOPASSWD: ALL

# sudo fallocate -l 18G /swapfile
# sudo chmod 600 /swapfile
# sudo mkswap /swapfile"
# sudo swapon /swapfile

# sudo $HOME/agent/scripts/setup.sh

# sudo fallocate -l 30G /swapfile
# sudo chmod 600 /swapfile
# sudo mkswap /swapfile
# sudo swapon /swapfile
