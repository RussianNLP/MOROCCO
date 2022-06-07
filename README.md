
# MOROCCO

MOdel ResOurCe COnsumption. Repository to evaluate Russian SuperGLUE models performance: inference speed, GPU RAM usage. Move from static text submissions with predictions to reproducible Docker-containers.

Each disc corresponds to baseline model, disc size is proportional to GPU RAM usage. By X axis there is model inference speed in records per second, by Y axis model score averaged by 9 Russian SuperGLUE tasks.

- Smaller models have higher inference speed. `rugpt3-small` processes ~200 records per second while `rugpt3-large` — ~60 records/second.
- `bert-multilingual` is a bit slower then `rubert*` due to worse Russian tokenizer. `bert-multilingual` splits text into more tokens, has to process larger batches.
- It is common that larger models show higher score but in our case `rugpt3-medium`, `rugpt3-large` perform worse then smaller `rubert*` models.
- `rugpt3-large` has more parameters then `rugpt3-medium` but is currently trained for less time and has lower score.

<img src="https://habrastorage.org/webt/wd/um/wt/wdumwtsu7bjxdhe1ot8hfclr3f8.png" />

## Papers

* <a href="https://arxiv.org/abs/2104.14314">MOROCCO: Model Resource Comparison Framework</a>
* <a href="https://arxiv.org/abs/2010.15925">RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark</a>

## How to measure my model performance using MOROCCO and submit it to Russian SuperGLUE leaderboard

## How to process user submission, add performance measurements to site
