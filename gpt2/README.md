# Build and use docker container for benchmarking GPT2 models

Build continaer from Dockerfile

```bash
cd MOROCCO
docker build -t gpt2_morocco:latest -f gpt2/Dockerfile .
```

Change all the variables in gpt2_stats.sh and gpt2_bench.sh:

- PYTHON_PATH is path to python environment with required dependencies installed
- VOL_PATH is path for docker volume, where gpt2 directory is located
- VOL_CHECKPOINT is directory with gpt2 model, path is relative to VOL_PATH
- MODEL_NAME is model identification name used in gpt2_stats.sh
- DEVICE is a string used to assign device through torch.device

OUTPUT_PATH gpt2_bench.sh and LOGS_PATH gpt2_stats.sh shall be identical

Specify model names in gpt2_bench.sh and launch the scripts
```bash
sh gpt2/scripts/gpt2_bench.sh
sh gpt2/scripts/gpt2_stats.sh
```

