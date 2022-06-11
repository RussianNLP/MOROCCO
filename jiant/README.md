
# Jiant baseline models. Train, infer, eval, build Docker container

Make Jiant baseline models reproducible:

- Use legacy <a href="https://github.com/nyu-mll/jiant-v1-legacy">Jiant 1.0</a>;
- Apply patch to support Russian SuperGLUE tasks `russian-superglue.patch`;
- Wrap Jiant into convenient `python main.py train|infer|eval`;
- Find optimal seeds. Original seeds <a href="https://russiansuperglue.com/leaderboard/2">used on leaderboard</a> were lost, try 5-10 random seeds for each model, use one with highest score on test;
- For each model for each task build Docker container, upload weights to Docker Hub.

## Dev notes, commands log

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
  --name default \
  --zone ru-central1-a \
  --network-interface subnet-name=default,nat-ip-version=ipv4 \
  --create-boot-disk image-folder-id=standard-images,image-family=ubuntu-2004-lts-gpu,type=network-hdd,size=50 \
  --cores=8 \
  --memory=96 \
  --gpus=1 \
  --ssh-key ~/.ssh/id_rsa.pub \
  --folder-name russian-superglue \
  --platform-id gpu-standard-v1 \
  --preemptible
```

Stop GPU instance, pay just for HDD storage.

```bash
yc compute instance stop --name default --folder-name russian-superglue
```

Start GPU instance.

```bash
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
ssh morocco mkdir jiant
scp main.py morocco:~/jiant
```

Sync back the code.

```bash
scp morocco:~/jiant/main.py .
```

Go to instance.

```bash
ssh morocco
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
# patch to support russian tasks
git clone https://github.com/nyu-mll/jiant-v1-legacy.git
cd jiant-v1-legacy
git checkout 61440732f4df2c54e68e59ead34b8656bf52af3b
git apply ../russian-superglue.patch


conda env create -f environment.yml -n main
conda activate main

# already installed deps via environment.yml
pip install --no-dependencies -e .

# https://github.com/allenai/specter/issues/27
pip install overrides==3.1.0
```

Fetch public Russian SuperGLUE data.

```bash
wget https://russiansuperglue.com/tasks/download -O combined.zip
unzip combined.zip
rm combined.zip
mkdir data
mv combined data/public
```

Upload private data.

```bash
scp -rv data/private morocco:~/jiant/data/
```

Train RuBERT on all tasks except RuCoS. Train wrapper removes all temporary files.

```bash
for task in rwsd parus rcb danetqa muserc russe terra
do
  python main.py train rubert $task exps/06 data/public --seed=3
done

# striped, keep only best snapshot weights
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

Infer trained model.

```bash
python main.py infer exps/06/rubert/ parus --batch-size=32 \
  < data/public/PARus/val.jsonl \
  > preds.jsonl

[2021-01-26 16:21:46] Reading items from stdin
[2021-01-26 16:21:46] Read 100 items
01/26 04:21:46 PM: PyTorch version 1.1.0 available.
[2021-01-26 16:21:46] Build tasks
[2021-01-26 16:21:53] Build model, load transformers pretrain
[2021-01-26 16:22:22] Load state '/home/yc-user/exps/06/rubert/parus/model.th'
01/26 04:22:31 PM: Loaded model state from /home/yc-user/exps/06/rubert/parus/model.th
[2021-01-26 16:22:31] Build mock task, infer via eval, batch_size=32
[2021-01-26 16:22:36] Writing preds to stdout

# preds.jsonl
{"idx": 0, "label": 1}
{"idx": 1, "label": 1}
{"idx": 2, "label": 1}
{"idx": 3, "label": 0}
{"idx": 4, "label": 1}
{"idx": 5, "label": 1}
{"idx": 6, "label": 0}
{"idx": 7, "label": 0}
...
```

No sudo docker.

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

Build docker container. First call takes long time, ~15 minutes. Docker caches build steps, icluding heavy `COPY transformers_cache exp/transformers_cache`. Second call take ~2 minutes, shares transformers_cache direcotry. Still takes time to copy build context, ~1.5GB per task.

```bash
python main.py docker-build exps/06/rubert terra rubert-terra
python main.py docker-build exps/06/rubert lidirus rubert-lidirus
```

Infer containerized model, provide test data as stdin, optionally define batch size, read preds from stdout. Docker container internally calls `main.py infer`.

```bash
docker run --gpus all --interactive --rm rubert-terra --batch-size 8 < data/public/TERRa/test.jsonl > preds.jsonl
docker run --gpus all --interactive --rm rubert-lidirus --batch-size 8 < data/public/LiDiRus/LiDiRus.jsonl > preds.jsonl

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
python main.py eval lidirus preds.jsonl data/private/LiDiRus/test_with_answers.jsonl
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

Select best performing. "!" - score < leaderboard score.

```
! exps/05/danetqa -> exps/rubert/danetqa
  exps/09/muserc -> exps/rubert/muserc
  exps/06/parus -> exps/rubert/parus
  exps/04/rcb -> exps/rubert/rcb
  exps/04/rucos -> exps/rubert/rucos
  exps/08/russe -> exps/rubert/russe
  exps/04/rwsd -> exps/rubert/rwsd
  exps/04/terra -> exps/rubert/terra
  exps/03/danetqa -> exps/rubert-conversational/danetqa
  exps/13/muserc -> exps/rubert-conversational/muserc
  exps/12/parus -> exps/rubert-conversational/parus
! exps/01/rcb -> exps/rubert-conversational/rcb
! exps/01/rucos -> exps/rubert-conversational/rucos
  exps/15/russe -> exps/rubert-conversational/russe
  exps/14/rwsd -> exps/rubert-conversational/rwsd
  exps/13/terra -> exps/rubert-conversational/terra
! exps/21/danetqa -> exps/bert-multilingual/danetqa
! exps/20/muserc -> exps/bert-multilingual/muserc
  exps/20/parus -> exps/bert-multilingual/parus
  exps/21/rcb -> exps/bert-multilingual/rcb
  exps/20/rucos -> exps/bert-multilingual/rucos
! exps/20/russe -> exps/bert-multilingual/russe
  exps/20/rwsd -> exps/bert-multilingual/rwsd
! exps/22/terra -> exps/bert-multilingual/terra
  exps/25/danetqa -> exps/rugpt3-small/danetqa
  exps/29/muserc -> exps/rugpt3-small/muserc
  exps/29/parus -> exps/rugpt3-small/parus
  exps/25/rcb -> exps/rugpt3-small/rcb
  exps/29/rucos -> exps/rugpt3-small/rucos
  exps/25/russe -> exps/rugpt3-small/russe
  exps/25/rwsd -> exps/rugpt3-small/rwsd
  exps/29/terra -> exps/rugpt3-small/terra
! exps/19/danetqa -> exps/rugpt3-medium/danetqa
  exps/24/muserc -> exps/rugpt3-medium/muserc
  exps/24/parus -> exps/rugpt3-medium/parus
  exps/19/rcb -> exps/rugpt3-medium/rcb
  exps/28/rucos -> exps/rugpt3-medium/rucos
  exps/19/russe -> exps/rugpt3-medium/russe
! exps/24/rwsd -> exps/rugpt3-medium/rwsd
  exps/28/terra -> exps/rugpt3-medium/terra
! exps/27/danetqa -> exps/rugpt3-large/danetqa
  exps/27/muserc -> exps/rugpt3-large/muserc
! exps/27/parus -> exps/rugpt3-large/parus
! exps/27/rcb -> exps/rugpt3-large/rcb
  exps/30/rucos -> exps/rugpt3-large/rucos
  exps/27/russe -> exps/rugpt3-large/russe
  exps/27/rwsd -> exps/rugpt3-large/rwsd
! exps/27/terra -> exps/rugpt3-large/terra
```

Generate token, login to Docker Hub.

```bash
docker login
```

Push to RussianNLP organization.

```bash
docker tag rubert-parus russiannlp/rubert-parus
docker push russiannlp/rubert-parus
```

Build and push all.

```bash
for model in rubert rubert-conversational bert-multilingual rugpt3-large rugpt3-medium rugpt3-small
do 
  for task in rwsd parus rcb danetqa muserc russe rucos terra lidirus
  do
    python main.py docker-build exps/$model $task $model-$task
    docker tag $model-$task russiannlp/$model-$task
    docker push russiannlp/$model-$task
  done
done
```

Infer and eval all.

```bash
declare -A titles
titles[danetqa]=DaNetQA
titles[lidirus]=LiDiRus
titles[muserc]=MuSeRC
titles[parus]=PARus
titles[rcb]=RCB
titles[rucos]=RuCoS
titles[russe]=RUSSE
titles[rwsd]=RWSD
titles[terra]=TERRa

for model in rubert-conversational rugpt3-medium rugpt3-small
do 
  mkdir -p preds/best/$model eval/best/$model
  for task in rwsd parus rcb danetqa muserc russe terra lidirus
  do
    docker run --gpus all --interactive --rm $model-$task --batch-size 8 < data/private/${titles[$task]}/test_with_answers.jsonl > preds/best/$model/$task.jsonl
    python main.py eval $task preds/best/$model/$task.jsonl data/private/${titles[$task]}/test_with_answers.jsonl > eval/best/$model/$task.json
  done
done
```
