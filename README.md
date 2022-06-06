
# MOROCCO

MOdel ResOurCe COnsumption. Repository to evaluate Russian SuperGLUE models performance: inference speed, GPU RAM usage. Move from static text submissions with predictions to reproducible Docker-containers.

Each disc corresponds to baseline model, disc size is proportional to GPU RAM usage. By X axis there is model inference speed in records per second, by Y axis model score averaged by 9 Russian SuperGLUE tasks.

- Smaller models have higher inference speed. `rugpt3-small` processes ~200 records per second while `rugpt3-large` — ~60 records/second.
- `bert-multilingual` is a bit slower then `rubert*` due to worse Russian tokenizer. `bert-multilingual` splits text into more tokens, has to process larger batches.
- It is common that larger models show higher score but in our case `rugpt3-medium`, `rugpt3-large` perform worse then smaller `rubert*` models.
- `rugpt3-large` has more parameters then `rugpt3-medium` but is currently trained for less time and has lower score.

<img src="https://habrastorage.org/webt/wd/um/wt/wdumwtsu7bjxdhe1ot8hfclr3f8.png" />

## How to measure my model performance using MOROCCO, submit to Russian SuperGLUE leaderboard

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

There are 9 tasks and 6 baseline models, so we built 9 * 6 containers: `rubert-danetqa`, `rubert-lidirus`, `rubert-muserc`, ..., `rugpt3-small-rwsd`, `rugpt3-small-terra`.

## Papers

* <a href="https://arxiv.org/abs/2104.14314">MOROCCO: Model Resource Comparison Framework</a>
* <a href="https://arxiv.org/abs/2010.15925">RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark</a>
