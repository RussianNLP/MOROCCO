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

Train RuBERT on all tasks except RuCoS. Train wrapper removes all temporary files.

```bash
for task in rwsd parus rcb danetqa muserc russe terra; \
  do python main.py train rubert $task ~/exps/06 ~/data/public --seed=3;
done

rubert/transformers_cache
rubert/transformers_cache/77e55469ea144f5cb185f025d9b21928dd446fc13e29a6bd19776059738229dd.71d8ad10edfcd7c68264ea03bc1b14e1f7b9c67affdfe8d6f96e1a6ce2c136ee.json
rubert/transformers_cache/3e164bb7396e401202e250721569fb407583681bb6ea0c34f431af622435a3
...
rubert/danetqa
rubert/danetqa/model.th
rubert/danetqa/log.log
rubert/danetqa/params.conf
rubert/rcb
rubert/rcb/model.th
rubert/rcb/log.log
rubert/rcb/params.conf
...
rubert/results.tsv
```

Upload weights to S3.

```bash
python main.py s3 sync ~/exps/06/rubert //exps/06
python main.py s3 sync ~/exps/12/rubert-conversational //exps/12
...
```

Download weights from S3. Fetch models with best scores.

```bash
python main.py s3 sync //exps/05/danetqa exps/rubert/danetqa
python main.py s3 sync //exps/09/muserc exps/rubert/muserc
python main.py s3 sync //exps/06/parus exps/rubert/parus
python main.py s3 sync //exps/04/rcb exps/rubert/rcb
python main.py s3 sync //exps/04/rucos exps/rubert/rucos
python main.py s3 sync //exps/08/russe exps/rubert/russe
python main.py s3 sync //exps/04/rwsd exps/rubert/rwsd
python main.py s3 sync //exps/04/terra exps/rubert/terra
python main.py s3 sync //exps/03/danetqa exps/rubert-conversational/danetqa
python main.py s3 sync //exps/13/muserc exps/rubert-conversational/muserc
python main.py s3 sync //exps/12/parus exps/rubert-conversational/parus
python main.py s3 sync //exps/01/rcb exps/rubert-conversational/rcb
python main.py s3 sync //exps/01/rucos exps/rubert-conversational/rucos
python main.py s3 sync //exps/15/russe exps/rubert-conversational/russe
python main.py s3 sync //exps/14/rwsd exps/rubert-conversational/rwsd
python main.py s3 sync //exps/13/terra exps/rubert-conversational/terra

python main.py s3 sync //exps/04/transformers_cache exps/rubert/transformers_cache
python main.py s3 sync //exps/13/transformers_cache exps/rubert-conversational/transformers_cache
```

No sudo docker.

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

Build docker container. First call takes long time, ~15 minutes. Docker caches build steps, icluding heavy `COPY transformers_cache exp/transformers_cache`. Second call take ~2 minutes, shares transformers_cache direcotry. Still takes time to copy build context, ~1.5GB per task.

```bash
python main.py docker build exps/rubert terra rubert-terra
python main.py docker build exps/rubert lidirus rubert-lidirus
```

Login to private registry.

```bash
python main.py docker login
```

Push model image. Tags image with cryptic Yandex.Cloud repo id.

```bash
python main.py push rubert-terra
python main.py push rubert-lidirus
```

Pull model image.

```bash
python main.py pull rubert-terra
python main.py pull rubert-lidirus
```

Infer containerized model, provide test data as stdin, optionally define batch size, read preds from stdout.

```bash
docker run --gpus all --interactive --rm rubert-terra --batch-size 8 < ~/data/public/TERRa/test.jsonl > preds.jsonl
docker run --gpus all --interactive --rm rubert-lidirus --batch-size 8 < ~/data/public/LiDiRus/LiDiRus.jsonl > preds.jsonl

[2021-01-18 11:49:31] Reading items from stdin
[2021-01-18 11:49:31] Read 1104 items
01/18 11:49:31 AM: PyTorch version 1.1.0 available.
[2021-01-18 11:49:31] Build tasks
[2021-01-18 11:49:32] Build model, load transformers pretrain
[2021-01-18 11:49:43] Load state 'exp/terra/model.th'
01/18 11:49:44 AM: Loaded model state from exp/terra/model.th
[2021-01-18 11:49:44] Build mock task, infer via eval, batch_size=8
/opt/conda/envs/jiant/lib/python3.6/site-packages/sklearn/metrics/classification.py:538: RuntimeWarning: invalid value encountered in double_scalars
  mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
[2021-01-18 11:50:06] Writing preds to stdout
{"idx": 0, "label": "not_entailment"}
{"idx": 1, "label": "entailment"}
{"idx": 2, "label": "entailment"}
{"idx": 3, "label": "entailment"}
{"idx": 4, "label": "not_entailment"}
{"idx": 5, "label": "not_entailment"}
{"idx": 6, "label": "entailment"}
{"idx": 7, "label": "entailment"}
{"idx": 8, "label": "entailment"}
...
```

Eval metrics.

```bash
python main.py eval lidirus preds.jsonl ~/data/private/LiDiRus/test_with_answers.jsonl
{
  "lex_sem": 0.23292650008324062,
  "lex_sem__Factivity;Quantifiers": 0.0,
  "lex_sem__Redundancy": 0.45652173913043476,
  "lex_sem__Lexical entailment;Factivity": 1.0,
  "lex_sem__Lexical entailment": 0.17620015937037806,
  "lex_sem__Symmetry/Collectivity": 0.0,
  "lex_sem__Lexical entailment;Quantifiers": 0.3333333333333333,
  "lex_sem__Quantifiers": 0.2689455496849596,
  "lex_sem__Named entities": 0.10050378152592121,
  "lex_sem__Morphological negation": -0.3239757976459313,
  ...
  "logic__Intervals/Numbers;Non-monotone": 0.0,
  "knowledge": 0.2389363008058508,
  "knowledge__World knowledge": 0.1516433124903911,
  "all_mcc": 0.21924343661028828,
  "accuracy": 0.5543478260869565
}
```
