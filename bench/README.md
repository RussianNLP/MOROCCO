
# Measure dockerized model performance: GPU RAM usage, inference speed

## Development

Notes on dev, commands log

Create GPU instance at Yandex.Cloud.

```bash
yc vpc network create --name default --folder-name russian-superglue
yc vpc subnet create \
  --name default \
  --network-name default \
  --range 192.168.0.0/24 \
  --zone ru-central1-a \
  --folder-name russian-superglue

yc compute instance create \
  --name default \
  --zone ru-central1-a \
  --network-interface subnet-name=default,nat-ip-version=ipv4 \
  --create-boot-disk image-folder-id=standard-images,image-family=ubuntu-2004-lts-gpu,type=network-hdd,size=100 \
  --cores=8 \
  --memory=96 \
  --gpus=1 \
  --ssh-key ~/.ssh/id_rsa.pub \
  --folder-name russian-superglue \
  --platform-id gpu-standard-v1 \
  --preemptible
```

Stop/start GPU instance.

```bash
yc compute instance stop --name default --folder-name russian-superglue
yc compute instance start --name default --folder-name russian-superglue
```

Drop GPU instance and network.

```bash
yc compute instance delete --name default --folder-name russian-superglue
yc vpc subnet delete --name default --folder-name russian-superglue
yc vpc network delete --name default --folder-name russian-superglue
```

Setup `~/.ssh/config`.

```bash
...
Host morocco
  Hostname 51.250.95.43
  User yc-user
...
```

Copy code to instance.

```bash
ssh morocco mkdir bench
scp main.py morocco:~/bench
```

Go to instance.

```bash
ssh morocco
```

Fetch public Russian SuperGLUE data.

```bash
wget https://russiansuperglue.com/tasks/download -O combined.zip
unzip combined.zip
rm combined.zip
mkdir data
mv combined data/public
```

Pull Jiant baseline model image for PARus.

```bash
docker pull russiannlp/rubert-parus
```

Estimate init time and model size. Run bench with 1 input record. Repeat 5 times for robust measure.

```bash
mkdir -p data/bench/rubert/parus
for index in 01 02 03 04 05
do
  python main.py bench russiannlp/rubert-parus data/public parus --input-size=1 --batch-size=1 > data/bench/rubert/parus/1_1_$index.jl
done
```

Bench all Jiant baseline models 5 times. Input size 2000 is optiomal not too short for robust time estimate, not to long to wait for process to finish. Batch size 32 is optimal, not to large fits in GPU RAM for all models, utilizes GPU for ~100% on all tasks.

```bash
input_size=1
batch_size=1
for model in rubert rubert-conversational bert-multilingual rugpt3-large rugpt3-medium rugpt3-smalldo
  for task in rwsd parus rcb danetqa muserc russe rucos terra lidirus
  do
    docker pull russiannlp/$model-$task
    mkdir -p data/bench/$model/$task
    for index in 01 02 03 04 05
	do
	  python main.py bench russiannlp/$model-$task data/public $task \
        --input-size=$input_size --batch-size=$batch_size \
        > data/bench/$model/$task/${input_size}_${batch_size}_${index}.jl
    done
  done
done
```
