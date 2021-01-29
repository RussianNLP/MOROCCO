# morocco

Repository to evaluate Russian SuperGLUE models performance: inference speed, GPU RAM usage.

<img src="https://habrastorage.org/webt/wd/um/wt/wdumwtsu7bjxdhe1ot8hfclr3f8.png" />

## Docker

We wrap baseline models into Docker containers. Container reads test data from stdin, writes predictions to stdout, has a single optional argument `--batch-size`. 

```bash
docker run --gpus all --interactive --rm bert-multilingual-parus --batch-size=32 \
  < ~/data/PARus/test.jsonl \
  > ~/pred.jsonl

# pred.jsonl
{"idx": 0, "label": 1}
{"idx": 1, "label": 1}
{"idx": 2, "label": 0}
{"idx": 3, "label": 1}
{"idx": 4, "label": 0}
{"idx": 5, "label": 0}
{"idx": 6, "label": 0}
{"idx": 7, "label": 0}
{"idx": 8, "label": 0}
{"idx": 9, "label": 1}
...
```

There are 9 tasks and 6 baseline models, so we built 9 * 6 containers: `rubert-danetqa`, `rubert-lidirus`, `rubert-muserc`, ..., `rugpt3-small-rwsd`, `rugpt3-small-terra`. We were not able to reproduce leaderboard scores exactly. 25% of containers show a bit lower score, for example leaderboard score for `rugpt3-large` on `rcb` is 0.417, out dockerized models gets 0.387.

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>danetqa</th>
      <th>muserc</th>
      <th>parus</th>
      <th>rcb</th>
      <th>rucos</th>
      <th>russe</th>
      <th>rwsd</th>
      <th>terra</th>
      <th>lidirus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rubert</th>
      <td>✅ 0.521 0.548</td>
      <td>❌ 0.639 0.635</td>
      <td>✅ 0.711/0.324 0.732/0.361</td>
      <td>✅ 0.574 0.630</td>
      <td>✅ 0.367/0.463 0.417/0.495</td>
      <td>✅ 0.320/0.314 0.331/0.329</td>
      <td>✅ 0.726 0.730</td>
      <td>✅ 0.669 0.714</td>
      <td>✅ 0.642 0.668</td>
      <td>✅ 0.191 0.219</td>
    </tr>
    <tr>
      <th>rubert-conversational</th>
      <td>✅ 0.500 0.519</td>
      <td>✅ 0.606 0.634</td>
      <td>✅ 0.687/0.278 0.700/0.290</td>
      <td>✅ 0.508 0.650</td>
      <td>❌ 0.452/0.484 0.446/0.479</td>
      <td>❌ 0.220/0.218 0.180/0.178</td>
      <td>✅ 0.729 0.734</td>
      <td>✅ 0.669 0.675</td>
      <td>✅ 0.640 0.648</td>
      <td>✅ 0.178 0.195</td>
    </tr>
    <tr>
      <th>bert-multilingual</th>
      <td>❌ 0.495 0.462</td>
      <td>❌ 0.624 0.606</td>
      <td>❌ 0.639/0.239 0.623/0.037</td>
      <td>✅ 0.528 0.536</td>
      <td>✅ 0.367/0.445 0.374/0.459</td>
      <td>✅ 0.290/0.290 0.296/0.294</td>
      <td>❌ 0.690 0.685</td>
      <td>✅ 0.669 0.675</td>
      <td>❌ 0.617 0.551</td>
      <td>❌ 0.189 0.063</td>
    </tr>
    <tr>
      <th>rugpt3-small</th>
      <td>✅ 0.438 0.483</td>
      <td>✅ 0.610 0.627</td>
      <td>✅ 0.653/0.221 0.702/0.292</td>
      <td>✅ 0.562 0.584</td>
      <td>✅ 0.356/0.473 0.431/0.479</td>
      <td>✅ 0.210/0.204 0.227/0.226</td>
      <td>✅ 0.570 0.581</td>
      <td>✅ 0.669 0.688</td>
      <td>✅ 0.488 0.563</td>
      <td>✅ -0.013 0.126</td>
    </tr>
    <tr>
      <th>rugpt3-medium</th>
      <td>✅ 0.468 0.501</td>
      <td>❌ 0.634 0.612</td>
      <td>✅ 0.706/0.308 0.717/0.315</td>
      <td>✅ 0.598 0.602</td>
      <td>✅ 0.372/0.461 0.437/0.502</td>
      <td>✅ 0.230/0.224 0.242/0.240</td>
      <td>✅ 0.642 0.660</td>
      <td>❌ 0.669 0.669</td>
      <td>✅ 0.505 0.606</td>
      <td>✅ 0.010 0.137</td>
    </tr>
    <tr>
      <th>rugpt3-large</th>
      <td>❌ 0.505 0.495</td>
      <td>❌ 0.604 0.596</td>
      <td>✅ 0.729/0.333 0.728/0.346</td>
      <td>❌ 0.584 0.566</td>
      <td>❌ 0.417/0.484 0.387/0.486</td>
      <td>✅ 0.210/0.202 0.243/0.241</td>
      <td>✅ 0.647 0.660</td>
      <td>✅ 0.636 0.669</td>
      <td>❌ 0.654 0.546</td>
      <td>❌ 0.231 0.200</td>
    </tr>
  </tbody>
</table>

*Scores on leaderboard versus scores of Docker containters*

## Performance

### GPU RAM

To measure model GPU RAM usage we run a container with a single record as input, measure maximum GPU RAM consumption, repeat procedure 5 times, take median value. `rubert`, `rubert-conversational`, `bert-multilingual`, `rugpt3-small` have approximately the same GPU RAM usage. `rugpt3-medium` is ~2 times larger than `rugpt3-small`, `rugpt3-large` is ~3 times larger.

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>danetqa</th>
      <th>muserc</th>
      <th>parus</th>
      <th>rcb</th>
      <th>rucos</th>
      <th>russe</th>
      <th>rwsd</th>
      <th>terra</th>
      <th>lidirus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rubert</th>
      <td>2.40</td>
      <td>2.40</td>
      <td>2.39</td>
      <td>2.39</td>
      <td>2.40</td>
      <td>2.39</td>
      <td>2.39</td>
      <td>2.39</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>rubert-conversational</th>
      <td>2.40</td>
      <td>2.40</td>
      <td>2.39</td>
      <td>2.39</td>
      <td>2.40</td>
      <td>2.39</td>
      <td>2.39</td>
      <td>2.39</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>bert-multilingual</th>
      <td>2.40</td>
      <td>2.40</td>
      <td>2.39</td>
      <td>2.39</td>
      <td>2.40</td>
      <td>2.39</td>
      <td>2.40</td>
      <td>2.39</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>rugpt3-small</th>
      <td>2.38</td>
      <td>2.38</td>
      <td>2.36</td>
      <td>2.37</td>
      <td>2.38</td>
      <td>2.36</td>
      <td>2.36</td>
      <td>2.37</td>
      <td>2.36</td>
    </tr>
    <tr>
      <th>rugpt3-medium</th>
      <td>4.41</td>
      <td>4.38</td>
      <td>4.39</td>
      <td>4.39</td>
      <td>4.38</td>
      <td>4.38</td>
      <td>4.41</td>
      <td>4.39</td>
      <td>4.39</td>
    </tr>
    <tr>
      <th>rugpt3-large</th>
      <td>7.49</td>
      <td>7.49</td>
      <td>7.50</td>
      <td>7.50</td>
      <td>7.49</td>
      <td>7.49</td>
      <td>7.51</td>
      <td>7.50</td>
      <td>7.50</td>
    </tr>
  </tbody>
</table>

*GPU RAM usage, GB*

### Inference speed

To measure inrefence speed we run a container with 2000 records as input, with batch size 32. On all tasks batch size 32 utilizes GPU almost at 100%. To estimate initialization time we run a container with input of size 1. Inference speed is (input size = 2000) / (total time - initialization time). We repeat procedure 5 times, take median value.

<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>danetqa</th>
      <th>muserc</th>
      <th>parus</th>
      <th>rcb</th>
      <th>rucos</th>
      <th>russe</th>
      <th>rwsd</th>
      <th>terra</th>
      <th>lidirus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>rubert</th>
      <td>118</td>
      <td>4</td>
      <td>1070</td>
      <td>295</td>
      <td>9</td>
      <td>226</td>
      <td>102</td>
      <td>297</td>
      <td>165</td>
    </tr>
    <tr>
      <th>rubert-conversational</th>
      <td>103</td>
      <td>4</td>
      <td>718</td>
      <td>289</td>
      <td>8</td>
      <td>225</td>
      <td>101</td>
      <td>302</td>
      <td>171</td>
    </tr>
    <tr>
      <th>bert-multilingual</th>
      <td>90</td>
      <td></td>
      <td>451</td>
      <td>194</td>
      <td>7</td>
      <td>164</td>
      <td>85</td>
      <td>195</td>
      <td>136</td>
    </tr>
    <tr>
      <th>rugpt3-small</th>
      <td>97</td>
      <td>4</td>
      <td>872</td>
      <td>289</td>
      <td></td>
      <td>163</td>
      <td>105</td>
      <td>319</td>
      <td>176</td>
    </tr>
    <tr>
      <th>rugpt3-medium</th>
      <td>45</td>
      <td>2</td>
      <td>270</td>
      <td>102</td>
      <td></td>
      <td>106</td>
      <td>70</td>
      <td>111</td>
      <td>106</td>
    </tr>
    <tr>
      <th>rugpt3-large</th>
      <td>27</td>
      <td>1</td>
      <td>137</td>
      <td>53</td>
      <td></td>
      <td>75</td>
      <td>49</td>
      <td>61</td>
      <td>69</td>
    </tr>
  </tbody>
</table>

*Inference speed, records per second*

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

Infer trained model.

```bash
python main.py infer ~/exps/06/rubert/ parus --batch-size=32 \
  < ~/data/public/PARus/val.jsonl \
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
python main.py s3 sync //exps/04/transformers_cache exps/rubert/transformers_cache
python main.py s3 sync //exps/03/danetqa exps/rubert-conversational/danetqa
python main.py s3 sync //exps/13/muserc exps/rubert-conversational/muserc
python main.py s3 sync //exps/12/parus exps/rubert-conversational/parus
python main.py s3 sync //exps/01/rcb exps/rubert-conversational/rcb
python main.py s3 sync //exps/01/rucos exps/rubert-conversational/rucos
python main.py s3 sync //exps/15/russe exps/rubert-conversational/russe
python main.py s3 sync //exps/14/rwsd exps/rubert-conversational/rwsd
python main.py s3 sync //exps/13/terra exps/rubert-conversational/terra
python main.py s3 sync //exps/13/transformers_cache exps/rubert-conversational/transformers_cache
python main.py s3 sync //exps/21/danetqa exps/bert-multilingual/danetqa
python main.py s3 sync //exps/20/parus exps/bert-multilingual/parus
python main.py s3 sync //exps/21/rcb exps/bert-multilingual/rcb
python main.py s3 sync //exps/20/rucos exps/bert-multilingual/rucos
python main.py s3 sync //exps/20/russe exps/bert-multilingual/russe
python main.py s3 sync //exps/20/rwsd exps/bert-multilingual/rwsd
python main.py s3 sync //exps/22/terra exps/bert-multilingual/terra
python main.py s3 sync //exps/22/transformers_cache exps/bert-multilingual/transformers_cache
python main.py s3 sync //exps/27/danetqa exps/rugpt3-large/danetqa
python main.py s3 sync //exps/27/muserc exps/rugpt3-large/muserc
python main.py s3 sync //exps/27/parus exps/rugpt3-large/parus
python main.py s3 sync //exps/27/rcb exps/rugpt3-large/rcb
python main.py s3 sync //exps/27/russe exps/rugpt3-large/russe
python main.py s3 sync //exps/27/rwsd exps/rugpt3-large/rwsd
python main.py s3 sync //exps/27/terra exps/rugpt3-large/terra
python main.py s3 sync //exps/27/transformers_cache exps/rugpt3-large/transformers_cache
python main.py s3 sync //exps/19/danetqa exps/rugpt3-medium/danetqa
python main.py s3 sync //exps/24/muserc exps/rugpt3-medium/muserc
python main.py s3 sync //exps/24/parus exps/rugpt3-medium/parus
python main.py s3 sync //exps/19/rcb exps/rugpt3-medium/rcb
python main.py s3 sync //exps/19/russe exps/rugpt3-medium/russe
python main.py s3 sync //exps/24/rwsd exps/rugpt3-medium/rwsd
python main.py s3 sync //exps/24/terra exps/rugpt3-medium/terra
python main.py s3 sync //exps/24/transformers_cache exps/rugpt3-medium/transformers_cache
python main.py s3 sync //exps/25/danetqa exps/rugpt3-small/danetqa
python main.py s3 sync //exps/25/muserc exps/rugpt3-small/muserc
python main.py s3 sync //exps/18/parus exps/rugpt3-small/parus
python main.py s3 sync //exps/25/rcb exps/rugpt3-small/rcb
python main.py s3 sync //exps/18/rucos exps/rugpt3-small/rucos
python main.py s3 sync //exps/25/russe exps/rugpt3-small/russe
python main.py s3 sync //exps/25/rwsd exps/rugpt3-small/rwsd
python main.py s3 sync //exps/18/terra exps/rugpt3-small/terra
python main.py s3 sync //exps/18/transformers_cache exps/rugpt3-small/transformers_cache
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

Infer containerized model, provide test data as stdin, optionally define batch size, read preds from stdout. Docker container internally calls `main.py infer`.

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

Upload preds.

```bash
python main.py s3 sync ~/preds //preds
```

Bench model 5 times. Input size 2000 is optiomal not too short for robust time estimate, not to long to wait for process to finish. Batch size 32 is optimal, not to large fits in GPU RAM for all models, utilizes GPU for ~100% on all tasks.

```bash
for index in 01 02 03 04 05; \
  do python main.py bench rubert-rucos ~/data rucos --input-size=2000 --batch-size=32 \
  > ~/benches/rubert/rwsd/2000_32_$index.jl; \
done
```

Upload bench measures.

```bash
python main.py s3 sync ~/benches //benches
```
