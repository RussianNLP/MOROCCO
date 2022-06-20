
# MOROCCO

MOdel ResOurCe COnsumption. Repository to evaluate Russian SuperGLUE models performance: inference speed, GPU RAM usage. Move from static text submissions with predictions to reproducible Docker-containers.

Each disc corresponds to <a href="jiant">Jiant baseline model</a>, disc size is proportional to GPU RAM usage. By X axis there is model inference speed in records per second, by Y axis model score averaged by 9 Russian SuperGLUE tasks.

- Smaller models have higher inference speed. `rugpt3-small` processes ~200 records per second while `rugpt3-large` — ~60 records/second.
- `bert-multilingual` is a bit slower than `rubert*` due to worse Russian tokenizer. `bert-multilingual` splits text into more tokens, has to process larger batches.
- It is common that larger models show higher score but in our case `rugpt3-medium`, `rugpt3-large` perform worse than smaller `rubert*` models.
- `rugpt3-large` has more parameters than `rugpt3-medium` but is currently trained for less time and has lower score.

<img width="529" alt="image" src="https://user-images.githubusercontent.com/153776/173176619-a76c313f-6e99-4b8d-a881-9e68b6466aad.png">

## Papers

* <a href="https://arxiv.org/abs/2104.14314">MOROCCO: Model Resource Comparison Framework</a>
* <a href="https://arxiv.org/abs/2010.15925">RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark</a>

## How to measure model performance using MOROCCO and submit it to Russian SuperGLUE leaderboard?

### Build Docker containers for each Russian SuperGLUE task

To benchmark model performance with MOROCCO use Docker, store model weights inside container, provide the following interface:

- Read test data from stdin;
- Write predictions to stdout;
- Handle `--batch-size` argument. MOROCCO runs container with `--batch-size=1` to estimate model size in GPU RAM.

```bash
docker pull russiannlp/rubert-parus
docker run --gpus all --interactive --rm \
  russiannlp/rubert-parus --batch-size 8 \
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

Refer to <a href="tfidf/">`tfidf/`</a> for minimal example and instructions on how to build Docker container. Minimal TF-IDF example runs on CPU, ignores `--batch-size` argument. Refer to <a href="jiant/">`jiant/`</a> for example on how to build GPU container.

Build containers for each Russian SuperGLUE task:

```bash
docker image ls

russiannlp/rubert-danetqa
russiannlp/rubert-lidirus
russiannlp/rubert-muserc
russiannlp/rubert-parus
russiannlp/rubert-rcb
russiannlp/rubert-rucos
russiannlp/rubert-russe
russiannlp/rubert-rwsd
russiannlp/rubert-terra
russiannlp/rugpt3-large-danetqa
russiannlp/rugpt3-large-lidirus
...
```

### Rent instance at Yandex Cloud

MOROCCO runs all benchmarks on the same hardware. We use <a href="https://cloud.yandex.ru/docs/compute/concepts/gpus">Yandex Cloud `gpu-standard-v1` instance</a>:

- NVIDIA® Tesla® V100 GPU with 32 GB GPU RAM
- 8 Intel Broadwell CPUs
- 96 GB RAM

We ask MOROCCO benchmark participants to rent the same instance at Yandex Cloud for their own expense. Current rent price is ~75 rubles/hour.

Create GPU instance using Yandex Cloud CLI:

- By default <a href="https://cloud.yandex.ru/docs/overview/concepts/quotas-limits">quota for number of GPU instances is zero</a>. <a href="https://console.cloud.yandex.ru/support/create-ticket">Create a ticket</a>, ask support to increase your quota to 1.
- Default HDD size is 50 GB, tweak `--create-boot-disk` to increase the size.
- `--preemptible` means that the instance is force stopped after 24 hours. Data stored on HDD is saved, all data in RAM is lost. Preemptible instance is cheaper, it costs ~75 rubles/hour.

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

Stop GPU instance, pay just for HDD storage. Start to continue experiments.

```bash
yc compute instance stop --name default --folder-name russian-superglue
yc compute instance start --name default --folder-name russian-superglue
```

Drop GPU instance, network and folder.

```bash
yc compute instance delete --name default --folder-name russian-superglue
yc vpc subnet delete --name default --folder-name russian-superglue
yc vpc network delete --name default --folder-name russian-superglue
yc resource-manager folder delete --name russian-superglue
```

### Produce benchmark logs

Use <a href="bench/main.py">`bench/main.py`</a> to collect CPU and GPU usage during container inference:

- Download <a href="https://russiansuperglue.com/tasks/">tasks data from Russian SuperGLUE site</a>, extract archive to `data/public/`;
- Increase/decrease `--input-size=2000` for optimal runtime. RuBERT processes 2000 PARus records in ~5 seconds, long enough to estimate inference speed;
- Increase/decrease `--batch-size=32` to max GPU RAM usage. RuBERT uses 100% GPU RAM on PARus with batch size 32;
- `main.py` calls `ps` and `nvidia-smi`, parses output, writes CPU and GPU usage to stdout, repeats 3 times per second.

```bash
python main.py bench russiannlp/rubert-parus data/public parus --input-size=2000 --batch-size=32 > 2000_32_01.jsonl

# data/public
data/public/LiDiRus
data/public/LiDiRus/LiDiRus.jsonl
data/public/MuSeRC
data/public/MuSeRC/test.jsonl
data/public/MuSeRC/val.jsonl
data/public/MuSeRC/train.jsonl
...

# 2000_32_01.jsonl
{"timestamp": 1655476624.532146, "cpu_usage": 0.953, "ram": 292663296, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476624.8558557, "cpu_usage": 0.767, "ram": 299151360, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476625.1793833, "cpu_usage": 0.767, "ram": 299151360, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476625.5032206, "cpu_usage": 0.83, "ram": 342458368, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476625.8275468, "cpu_usage": 0.728, "ram": 349483008, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476626.1513274, "cpu_usage": 0.762, "ram": 341012480, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476626.4759278, "cpu_usage": 0.762, "ram": 341012480, "gpu_usage": null, "gpu_ram": null}
...
{"timestamp": 1655476632.3156314, "cpu_usage": 0.775, "ram": 1693970432, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476632.6450512, "cpu_usage": 0.78, "ram": 1728303104, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476632.975281, "cpu_usage": 0.728, "ram": 1758257152, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476633.3079898, "cpu_usage": 0.8, "ram": 1758818304, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476633.6325083, "cpu_usage": 0.808, "ram": 1787203584, "gpu_usage": null, "gpu_ram": null}
{"timestamp": 1655476633.9611752, "cpu_usage": 0.774, "ram": 1199480832, "gpu_usage": 0.0, "gpu_ram": 12582912}
{"timestamp": 1655476634.413833, "cpu_usage": 0.78, "ram": 1324830720, "gpu_usage": 0.0, "gpu_ram": 326107136}
{"timestamp": 1655476634.7563012, "cpu_usage": 0.727, "ram": 1331073024, "gpu_usage": 0.0, "gpu_ram": 393216000}
{"timestamp": 1655476635.0970583, "cpu_usage": 0.73, "ram": 1334509568, "gpu_usage": 0.0, "gpu_ram": 405798912}
{"timestamp": 1655476635.4380798, "cpu_usage": 0.74, "ram": 1387737088, "gpu_usage": 0.02, "gpu_ram": 433061888}
{"timestamp": 1655476635.7793305, "cpu_usage": 0.696, "ram": 1425448960, "gpu_usage": 0.0, "gpu_ram": 445644800}
{"timestamp": 1655476636.1234272, "cpu_usage": 0.698, "ram": 1447387136, "gpu_usage": 0.0, "gpu_ram": 451936256}
{"timestamp": 1655476636.4652247, "cpu_usage": 0.704, "ram": 1506942976, "gpu_usage": 0.0, "gpu_ram": 462422016}
{"timestamp": 1655476636.8055842, "cpu_usage": 0.668, "ram": 1542393856, "gpu_usage": 0.02, "gpu_ram": 485490688}
{"timestamp": 1655476637.146097, "cpu_usage": 0.673, "ram": 1587482624, "gpu_usage": 0.0, "gpu_ram": 495976448}
{"timestamp": 1655476637.4880967, "cpu_usage": 0.678, "ram": 1635229696, "gpu_usage": 0.01, "gpu_ram": 512753664}
{"timestamp": 1655476637.8288727, "cpu_usage": 0.641, "ram": 1664548864, "gpu_usage": 0.01, "gpu_ram": 523239424}
...
```

Produce benchmark logs for each task:

- Benchmark with `--input-size=1`, `--batch-size=1`. This way MOROCCO estimates model init time and model size in GPU RAM. We assume that 1 record takes almost no time to process and almost no space in GPU RAM. So all run time is init time and max GPU RAM usage is model size;
- Benchmark with `--input-size=X`, `--batch-size=Y` where `X > 1`. Choose such `X` so that model takes at least several seconds to process input. Otherwise the inference speed estimate is not robust. Choose such `Y` so that model still fits in GPU RAM, maximize GPU utilization, inferefence speed;
- Repeat every measurement 5 times for better median estimates;
- Save logs to `logs/$task/${input_size}_${batch_size}_${index}.jsonl` files. Do not change path pattern, `main.py plot|stats` parse file path to get task, input and batch sizes.

```bash
input_size=2000
batch_size=32
model=russiannlp/rubert
for task in rwsd parus rcb danetqa muserc russe rucos terra lidirus
do
  mkdir -p logs/$task
  for index in 01 02 03 04 05
    do
	  python main.py bench $model-$task data/public $task \
        --input-size=$input_size --batch-size=$batch_size \
        > logs/$task/${input_size}_${batch_size}_${index}.jsonl
  done
done

# Repeat with
# input_size=2000
# batch_size=32

```

Final `logs/` structure should have 9 * 5 * 2 files:

```bash
logs/
logs/danetqa
logs/danetqa/1_1_01.jsonl
logs/danetqa/1_1_02.jsonl
logs/danetqa/1_1_03.jsonl
logs/danetqa/1_1_04.jsonl
logs/danetqa/1_1_05.jsonl
logs/danetqa/2000_32_01.jsonl
logs/danetqa/2000_32_02.jsonl
logs/danetqa/2000_32_03.jsonl
logs/danetqa/2000_32_04.jsonl
logs/danetqa/2000_32_05.jsonl
logs/lidirus
logs/lidirus/1_1_01.jsonl
logs/lidirus/1_1_02.jsonl
logs/lidirus/1_1_03.jsonl
...
```

### Use logs to estimate model RAM usage and inference speed, make sure benchmark logs are correct

Use `main.py plot` to plot log records:

```bash
pip install pandas matplotlib
mkdir -p plots

python main.py plot logs/parus/*.jsonl plots/parus.png
```

Examine the plot, make sure benchmark logs are correct:

- Look at `cpu_usage` plot. 4 runs with `--input-size=1` take ~17 sec, 1 outlier run takes ~24 sec. MOROCCO computes median time, so final init time estimate is 17 sec. RuBERT Jiant implementation takes a long time to start. All runs with `--input-size=2000` take ~20 sec. Inference speed estimate is 2000 / (20 - 17);
- Make sure outliers do not affect final estimates. Otherwise remove log file, rerun benchmark;
- Look at `gpu_ram` plot. Maximum GPU RAM usage with `--input-size=1` is ~2.4 GB, MOROCCO treats it as RuBERT model size. GPU RAM usage with `--batch-size=32` is just a tiny bit larger;
- Look at `gpu_usage` plot. Minimal GPU utilization is 82%, could increase batch size to make it close to 100%.

<img width="597" alt="image" src="https://user-images.githubusercontent.com/153776/174425062-d3762481-9e13-43d1-9b9d-d48ff6593a9c.png">

Use `main.py stats` to process logs, get performance estimates. Make sure estimates match plots:

- `gpu_ram` is ~2.4 GB, matches maximum GPU RAM usage on `gpu_ram` plot.
- `rps` is close to 2000 / (20 - 17), matches `cpu_usage` plot

```bash
python main.py stats logs/parus/*.jsonl >> stats.jsonl

# stats.jsonl
{"task": "parus", "gpu_ram": 2.3701171875, "rps": 604.8896607058683}
```

Repeat for all tasks:

```bash
rm -f stats.jsonl
for task in rwsd parus rcb danetqa muserc russe rucos terra lidirus
do
  python main.py plot logs/$task/*.jsonl plots/$task.png
  python main.py stats logs/$task/*.jsonl >> stats.jsonl
done
```

### Send logs archive to Russian SuperGLUE site

Archive logs into `logs.zip`:

```bash
sudo apt install zip
zip logs.zip -r logs
```

Submit `logs.zip` to Russian SuperGLUE site. *WARN* Submission form is not yet implemented

Notice `stats.jsonl` is not submitted. Russian SuperGLUE organizers use logs, compute stats internally.

### Optionally upload Docker containers with your model to Docker Hub, sent links to Russian SuperGLUE site

To make model publicly available upload containers to Docker Hub. To keep model private just skip this step.

Create account on Docker Hub. Go to account settings, generate token, login, upload container. Change `rubert` and `russiannlp` to your model and account name:

```bash
docker login

docker tag rubert-parus russiannlp/rubert-parus
docker push russiannlp/rubert-parus
```

Repeat for all tasks:

```bash
for task in rwsd parus rcb danetqa muserc russe rucos terra lidirus
do
  docker tag rubert-$task russiannlp/rubsert-$task
  docker push russiannlp/rubsert-$task
done
```

Submit links to Russian SuperGLUE site. *WARN* Submission form is not yet implemented

```bash
https://hub.docker.com/r/russiannlp/rubert-rwsd
https://hub.docker.com/r/russiannlp/rubert-parus
...
```

### Q&A

#### `main.py bench` raises `pid not found` error

Just relaunch `main.py`, have no idea why error happens.

#### What if I use the same base model for all tasks, why duplicate it in 9 containers?

Imagine two Dockerfiles:

```dockerfile
# Dockerfile.parus
ADD model.th .
ADD infer.py .

RUN python infer.py model.th parus
```

```dockerfile
# Dockerfile.terra
ADD model.th .
ADD infer.py .

RUN python infer.py model.th terra
```

Docker shares `model.th` and `infer.py` between `terra` and `parus` containers. Learn more about <a href="https://dockerlabs.collabnix.com/beginners/dockerfile/Layering-Dockerfile.html">Layering in Docker</a>. So even if `model.th` is a large file, only first container build is slow.
