# morocco

## Development

Notes on dev, commands log.

Create GPU instance at Yandex.Cloud. 500GB HDD to train multiple models, Jiant dumps number of checkpoints, uses lots of storage.

```bash
yc vpc network create --name default --folder-name russian-superglue
yc vpc subnet create \
  --name default \
  --network-name default \
  --range 192.168.0.0/24 \
  --zone ru-central1-a \
  --folder-name russian-superglue

yc compute instance create \
  --name gpu \
  --zone ru-central1-a \
  --network-interface subnet-name=default,nat-ip-version=ipv4 \
  --create-boot-disk image-folder-id=standard-images,image-family=ubuntu-2004-lts-gpu,type=network-hdd,size=500 \
  --cores=8 \
  --memory=96 \
  --gpus=1 \
  --ssh-key ~/.ssh/id_rsa.pub \
  --folder-name russian-superglue \
  --platform-id gpu-standard-v1 \
  --preemptible
```

Drop GPU instance and network.

```bash
yc compute instance delete --name gpu --folder-name russian-superglue
yc vpc subnet delete --name default --folder-name russian-superglue
yc vpc network delete --name default --folder-name russian-superglue
```

Copy data and code to instance.

```bash
tar czvf data.tar.gz -C ~/proj/russian-superglue data
scp -v data.tar.gz gpu:~
ssh gpu tar xzvf data.tar.gz
ssh gpu rm data.tar.gz
rm data.tar.gz

ssh gpu mkdir morocco
scp {main.ipynb,main.py} gpu:~/morocco
```

Sync back the code.

```bash
scp 'gpu:~/morocco/{main.ipynb,main.py}' .
```

Setup instance.

```bash
sudo locale-gen ru_RU.UTF-8
sudo apt-get update
sudo apt-get install -y \
  python3-pip \
  python-is-python3 \
  screen

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && sudo bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/anaconda3 \
  && rm Miniconda3-latest-Linux-x86_64.sh
/opt/anaconda3/bin/conda init
# relogin
```

Init conda env with Russian fork of Jiant. Takes ~10 minutes.

```bash
git clone https://github.com/kuk/jiant-v1-legacy.git
cd jiant-v1-legacy
git checkout russian-superglue

conda env create -f environment.yml -n main
conda activate main

# already installed deps via environment.yml
pip install --no-dependencies -e .
```

Install, launch Jupyter.

```bash
pip install notebook

screen
jupyter notebook \
  --no-browser \
  --allow-root \
  --port=8888 \
  --NotebookApp.token='' \
  --NotebookApp.password=''
```

Reverse tunnel Jupyter. Go to http://localhost:8888/notebooks/

```bash
ssh -Nf gpu -L 8888:localhost:8888
```
