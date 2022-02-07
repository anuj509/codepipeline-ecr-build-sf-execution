sudo apt-get install software-properties-common

sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6 libpython3.6

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 2
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

sudo rm /usr/bin/python3
sudo ln -s python3.6 /usr/bin/python3

#ref
#http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/