
# MOROCCO

MOdel ResOurCe COnsumption. Repository to evaluate Russian SuperGLUE models performance: inference speed, GPU RAM usage. Move from static text submissions with predictions to reproducible Docker-containers.

Each disc corresponds to <a href="jiant">Jiant baseline model</a>, disc size is proportional to GPU RAM usage. By X axis there is model inference speed in records per second, by Y axis model score averaged by 9 Russian SuperGLUE tasks.

- Smaller models have higher inference speed. `rugpt3-small` processes ~200 records per second while `rugpt3-large` — ~60 records/second.
- `bert-multilingual` is a bit slower then `rubert*` due to worse Russian tokenizer. `bert-multilingual` splits text into more tokens, has to process larger batches.
- It is common that larger models show higher score but in our case `rugpt3-medium`, `rugpt3-large` perform worse then smaller `rubert*` models.
- `rugpt3-large` has more parameters then `rugpt3-medium` but is currently trained for less time and has lower score.

<img width="529" alt="image" src="https://user-images.githubusercontent.com/153776/173176619-a76c313f-6e99-4b8d-a881-9e68b6466aad.png">

## Papers

* <a href="https://arxiv.org/abs/2104.14314">MOROCCO: Model Resource Comparison Framework</a>
* <a href="https://arxiv.org/abs/2010.15925">RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark</a>

## How to measure model performance using MOROCCO and submit it to Russian SuperGLUE leaderboard?

### Build Docker containers for each Russian SuperGLUE task

To benchmark model performance with MOROCCO it should have the following interface:

- Use Docker. Store model weights inside container, fix hyperparameters in Dockerfile
- Read test data from stdin
- Write predictions to stdout
- Handle `--batch-size` argument. MOROCCO runs container with `--batch-size=1` to estimate model size in GPU RAM, exclude data size.

```bash
docker pull russiannlp/rubert-parus
docker run --gpus all --interactive --rm \
  russiannlp/rubert-terra --batch-size 8 \
  < TERRa/test.jsonl \
  > preds.jsonl

# TERRa/test.jsonl
{"premise": "Гвардейцы подошли к грузовику, ...", "hypothesis": "Гвардейцы подошли к сломанному грузовику.", "idx": 0}
{"premise": "\"К настоящему моменту число ...", "hypothesis": "Березовский открывает аккаунты во всех соцсетях.", "idx": 1}
...

# preds.jsonl
{"idx": 0, "label": "entailment"}
{"idx": 1, "label": "entailment"}
...
```

Refer to <a href="tfidf/">`tfidf/`</a> for minimal example and instructions how to build Docker container.

### Rent instance at Yandex Cloud

All benchmarks run on the same hardware. We use <a href="https://cloud.yandex.ru/docs/compute/concepts/gpus">Yandex Cloud `gpu-standard-v1` instance</a>:

- NVIDIA® Tesla® V100 GPU with 32 GB GPU RAM
- 8 Intel Broadwell CPUs
- 96 GB RAM

We ask MOROCCO benchmark participants to rent the same instance at Yandex Cloud for their own expence. Currently rent price is ~75 rubles/hour.

Create GPU instance using Yandex Cloud CLI:

- By default <a href="https://cloud.yandex.ru/docs/overview/concepts/quotas-limits">quota for number of GPU instances is zero</a>. <a href="https://console.cloud.yandex.ru/support/create-ticket">Create a ticket</a>, ask support to increase your quota to 1.
- Default HDD size is 50 GB, tweak `--create-boot-disk` to increase the size.
- `--preemptible` means that your instance is force stopped after 24 hours. Data stored on HDD is saved, all data in RAM is lost. Currently such instance costs ~75 rubles/hour.

```bash
yc resource-manager folder create --name russian-superglue
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

### Produce benchmark logs

### Use logs to estimate RAM usage and inference speed, make sure benchmark logs are correct

### Send logs archive to Russian SuperGLUE site

### Optionally upload Docker containers with your model to Docker Hub, sent links to Russian SuperGLUE site

## How to process user submission, add performance measurements to site
