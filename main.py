
import re
import json
from collections import (
    Counter,
    defaultdict
)
from random import (
    seed,
    random,
    choice,
    sample,
)
from dataclasses import dataclass
from math import ceil
from os import listdir
from os.path import (
    join,
    exists
)
from math import sqrt
import statistics

import pandas as pd

from matplotlib import patches
from matplotlib import pyplot as plt


DANETQA = 'danetqa'
LIDIRUS = 'lidirus'
MUSERC = 'muserc'
PARUS = 'parus'
RCB = 'rcb'
RUCOS = 'rucos'
RUSSE = 'russe'
RWSD = 'rwsd'
TERRA = 'terra'
TASKS = [
    DANETQA,
    MUSERC,
    PARUS,
    RCB,
    RUCOS,
    RUSSE,
    RWSD,
    TERRA,
    LIDIRUS,
]
TASK_TITLES = {
    DANETQA: 'DaNetQA',
    LIDIRUS: 'LiDiRus',
    MUSERC: 'MuSeRC',
    PARUS: 'PARus',
    RCB: 'RCB',
    RUCOS: 'RuCoS',
    RUSSE: 'RUSSE',
    RWSD: 'RWSD',
    TERRA: 'TERRa',
}

RUBERT = 'rubert'
RUBERT_CONVERSATIONAL = 'rubert-conversational'
BERT_MULTILINGUAL = 'bert-multilingual'

RUGPT3_LARGE = 'rugpt3-large'
RUGPT3_MEDIUM = 'rugpt3-medium'
RUGPT3_SMALL = 'rugpt3-small'

MODELS = [
    RUBERT,
    RUBERT_CONVERSATIONAL,
    BERT_MULTILINGUAL,

    RUGPT3_SMALL,
    RUGPT3_MEDIUM,
    RUGPT3_LARGE,
]

HUMAN = 'human'
TFIDF = 'tfidf'

MODEL_HUB_NAMES = {
    RUBERT_CONVERSATIONAL: 'DeepPavlov/rubert-base-cased-conversational',
    RUBERT: 'DeepPavlov/rubert-base-cased',
    BERT_MULTILINGUAL: 'bert-base-multilingual-cased',

    RUGPT3_LARGE: 'sberbank-ai/rugpt3large_based_on_gpt2',
    RUGPT3_MEDIUM: 'sberbank-ai/rugpt3medium_based_on_gpt2',
    RUGPT3_SMALL: 'sberbank-ai/rugpt3small_based_on_gpt2',
}

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

MODEL_HUB_SIZES = {
    RUBERT: 679 * MB,
    RUBERT_CONVERSATIONAL: 679 * MB,
    BERT_MULTILINGUAL: 682 * MB,

    RUGPT3_SMALL: 526 * MB,
    RUGPT3_MEDIUM: 1.7 * GB,
    RUGPT3_LARGE: 3.0 * GB,
}


JIANT_DATA_DIR = 'jiant/data'
JIANT_PREDS_DIR = join(JIANT_DATA_DIR, 'preds')
JIANT_EVAL_DIR = join(JIANT_DATA_DIR, 'eval')

DATA_DIR = '../data'

GRID = 'grid'
BEST = 'best'

PRIVATE = 'private'
PUBLIC = 'public'

TRAIN = 'train'
TEST = 'test'
VAL = 'val'

JSON = '.json'

JIANT_BENCH_DIR = 'bench/data/jiant'


#####
#
#  IO
#
#####


def load_lines(path):
    with open(path) as file:
        for line in file:
            yield line.rstrip('\n')


def parse_jl(lines):
    for line in lines:
        yield json.loads(line)


def load_jl(path):
    lines = load_lines(path)
    return parse_jl(lines)


def parse_tsv(lines, sep='\t'):
    for line in lines:
        yield line.split(sep)


def load_text(path):
    with open(path) as file:
        return file.read()


parse_json = json.loads


def load_json(path):
    text = load_text(path)
    return parse_json(text)


#######
#
#   LEADERBOARD
#
#######


LEADERBOARD_LINES = '''
Rank	Name	Team	Link	Score	LiDiRus	RCB	PARus	MuSeRC	TERRa	RUSSE	RWSD	DaNetQA	RuCoS
1	HUMAN BENCHMARK	AGI NLP		0.811	0.626	0.68 / 0.702	0.982	0.806 / 0.42	0.92	0.805	0.84	0.915	0.93 / 0.89
2	RuBERT plain	DeepPavlov		0.521	0.191	0.367 / 0.463	0.574	0.711 / 0.324	0.642	0.726	0.669	0.639	0.32 / 0.314
3	RuGPT3Large	sberdevices		0.505	0.231	0.417 / 0.484	0.584	0.729 / 0.333	0.654	0.647	0.636	0.604	0.21 / 0.202
4	RuBERT conversational	DeepPavlov		0.5	0.178	0.452 / 0.484	0.508	0.687 / 0.278	0.64	0.729	0.669	0.606	0.22 / 0.218
5	Multilingual Bert	DeepPavlov		0.495	0.189	0.367 / 0.445	0.528	0.639 / 0.239	0.617	0.69	0.669	0.624	0.29 / 0.29
6	RuGPT3Medium	sberdevices		0.468	0.01	0.372 / 0.461	0.598	0.706 / 0.308	0.505	0.642	0.669	0.634	0.23 / 0.224
7	RuGPT3Small	sberdevices		0.438	-0.013	0.356 / 0.473	0.562	0.653 / 0.221	0.488	0.57	0.669	0.61	0.21 / 0.204
8	Baseline TF-IDF1.1	AGI NLP		0.434	0.06	0.301 / 0.441	0.486	0.587 / 0.242	0.471	0.57	0.662	0.621	0.26 / 0.252
'''.strip().splitlines()

LEADERBOARD_RENAMES = {
    'HUMAN BENCHMARK': HUMAN,
    'RuBERT plain': RUBERT,
    'RuGPT3Large': RUGPT3_LARGE,
    'RuBERT conversational': RUBERT_CONVERSATIONAL,
    'Multilingual Bert': BERT_MULTILINGUAL,
    'RuGPT3Medium': RUGPT3_MEDIUM,
    'RuGPT3Small': RUGPT3_SMALL,
    'Baseline TF-IDF1.1': TFIDF,

    'LiDiRus': LIDIRUS,
    'RCB': RCB,
    'PARus': PARUS,
    'MuSeRC': MUSERC,
    'TERRa': TERRA,
    'RUSSE': RUSSE,
    'RWSD': RWSD,
    'DaNetQA': DANETQA,
    'RuCoS': RUCOS,
}


def parse_leaderboard_score(value, sep=' / '):
    if sep in value:
        # 0.301 / 0.441
        parts = value.split(sep)
        first, second = map(float, parts)
    else:
        first, second = float(value), None
    return Score(first, second)


def parse_leaderboard(records, name_offset=1, scores_offset=5):
    header = next(records)
    tasks = [LEADERBOARD_RENAMES[_] for _ in header[scores_offset:]]
    for record in records:
        model = LEADERBOARD_RENAMES[record[name_offset]]
        scores = [parse_leaderboard_score(_) for _ in record[scores_offset:]]
        for task, score in zip(tasks, scores):
            yield model, task, score


def find_leaderboard_score(model, task, leaderboard):
    for leaderboard_model, leaderboard_task, score in leaderboard:
        if leaderboard_model == model and leaderboard_task == task:
            return score


#######
#
#  SCORE
#
#####


@dataclass
class Score:
    first: float
    second: float = None


TASK_METRICS = {
    DANETQA: 'accuracy',
    MUSERC: ('ans_f1', 'em'),
    PARUS: 'accuracy',
    RCB: ('f1', 'accuracy'),
    RUCOS: ('f1', 'em'),
    RUSSE: 'accuracy',
    RWSD: 'acc',
    TERRA: 'accuracy',
    LIDIRUS: 'all_mcc',
}


def metrics_score(task, metrics):
    key = TASK_METRICS[task]
    if isinstance(key, tuple):
        first, second = key
        first, second = metrics[first], metrics[second]
    else:
        first = metrics[key]
        second = None
    return Score(first, second)


def score_value(score):
    if score.second:
        return (score.first + score.second) / 2
    return score.first


def format_score(score, digits=3, sep=' / '):
    pattern = '{:.%df}' % digits
    if score.second:
        return '{first}{sep}{second}'.format(
            first=pattern.format(score.first),
            sep=sep,
            second=pattern.format(score.second)
        )
    return pattern.format(score.first)


#####
#
#  TASK
#
######


def task_path(task, access, split, dir=DATA_DIR):
    title = TASK_TITLES[task]

    name = split
    if split == TEST:
        if access == PUBLIC and task == LIDIRUS:
            name = title
        elif access == PRIVATE:
            name = 'test_with_answers'

    return join(dir, access, title, f'{name}.jsonl')


def load_task(task, access, split):
    path = task_path(task, access, split)
    return load_jl(path)


######
#
#   JIANT GRID
#
######


@dataclass
class JiantGridConf:
    id: str
    model: str
    seed: int


JIANT_GRID_CONFS = [
    JiantGridConf('01', RUBERT_CONVERSATIONAL, seed=111),
    JiantGridConf('02', RUBERT_CONVERSATIONAL, seed=2),
    JiantGridConf('03', RUBERT_CONVERSATIONAL, seed=3),

    JiantGridConf('04', RUBERT, seed=1),
    JiantGridConf('05', RUBERT, seed=2),
    JiantGridConf('06', RUBERT, seed=3),
    JiantGridConf('07', RUBERT, seed=4),
    JiantGridConf('08', RUBERT, seed=5),
    JiantGridConf('09', RUBERT, seed=6),
    JiantGridConf('10', RUBERT, seed=7),
    JiantGridConf('11', RUBERT, seed=8),

    JiantGridConf('12', RUBERT_CONVERSATIONAL, seed=4),
    JiantGridConf('13', RUBERT_CONVERSATIONAL, seed=5),
    JiantGridConf('14', RUBERT_CONVERSATIONAL, seed=6),
    JiantGridConf('15', RUBERT_CONVERSATIONAL, seed=7),
    JiantGridConf('16', RUBERT_CONVERSATIONAL, seed=8),

    JiantGridConf('17', RUBERT, seed=9),

    JiantGridConf('18', RUGPT3_SMALL, seed=1),
    JiantGridConf('19', RUGPT3_MEDIUM, seed=1),

    JiantGridConf('20', BERT_MULTILINGUAL, seed=1),
    JiantGridConf('21', BERT_MULTILINGUAL, seed=2),
    JiantGridConf('22', BERT_MULTILINGUAL, seed=3),

    JiantGridConf('23', RUGPT3_MEDIUM, seed=2),
    JiantGridConf('24', RUGPT3_MEDIUM, seed=3),

    JiantGridConf('25', RUGPT3_SMALL, seed=2),
    JiantGridConf('26', RUGPT3_SMALL, seed=3),

    JiantGridConf('27', RUGPT3_LARGE, seed=1),
    JiantGridConf('28', RUGPT3_MEDIUM, seed=4),
    JiantGridConf('29', RUGPT3_SMALL, seed=4),
    JiantGridConf('30', RUGPT3_LARGE, seed=2),
]


def select_top_conf(task, model, grid_task_scores, confs=JIANT_GRID_CONFS):
    id, max_score = None, None
    id_confs = {_.id: _ for _ in confs}
    for conf_id, conf_task, score in grid_task_scores:
        score = score_value(score)
        conf = id_confs[conf_id]
        if conf.model == model and conf_task == task and (max_score is None or max_score < score):
            id = conf_id
            max_score = score
    return id


def find_grid_score(id, task, grid_task_scores):
    for conf_id, conf_task, score in grid_task_scores:
        if conf_id == id and conf_task == task:
            return score


#######
#
#  BENCH
#
#########


@dataclass
class BenchRecord:
    timestamp: float
    cpu_usage: float = None
    ram: int = None
    gpu_usage: float = None
    gpu_ram: int = None


def load_bench(path):
    items = load_jl(path)
    for item in items:
        yield BenchRecord(**item)


#######
#  REGISTRY
#######


@dataclass
class BenchRegistryRecord:
    dir: str
    model: str
    task: str
    input_size: int
    batch_size: int
    index: int


def list_bench_registry(dir):
    for model in listdir(dir):
        for task in listdir(join(dir, model)):
            for filename in listdir(join(dir, model, task)):
                match = re.match(r'(\d+)_(\d+)_(\d+)\.jl', filename)
                if match:
                    input_size, batch_size, index = map(int, match.groups())
                    yield BenchRegistryRecord(
                        dir, model, task,
                        input_size, batch_size, index
                    )


def match_bench_registry_record(record, **kwargs):
    for key, value in kwargs.items():
        if not isinstance(value, (tuple, list)):
            value = [value]
        if getattr(record, key) not in value:
            return False
    return True


def query_bench_registry(records, **kwargs):
    for record in records:
        if match_bench_registry_record(record, **kwargs):
            yield record


def registry_bench_path(dir, model, task, input_size, batch_size, index=1):
    return join(dir, model, task, f'{input_size}_{batch_size}_{index:02d}.jl')


def load_registry_bench(record):
    path = registry_bench_path(
        record.dir, record.model, record.task,
        record.input_size, record.batch_size, record.index
    )
    return list(load_bench(path))


#######
#   STATS
#####


@dataclass
class BenchStats:
    total_time: int
    gpu_time: int
    max_gpu_ram: int


def safe_max(values):
    values = list(values)
    if values:
        return max(values)


def bench_stats(records, gpu_usage_treshold=0.1):
    total_time = (
        records[-1].timestamp
        - records[0].timestamp
    )
    max_gpu_ram = safe_max(
        _.gpu_ram
        for _ in records
        if _.gpu_ram
    )

    previous = None
    gpu_time = 0
    for record in records:
        if (
                record.gpu_usage
                and record.gpu_usage >= gpu_usage_treshold
                and previous
        ):
            gpu_time += (record.timestamp - previous)
        previous = record.timestamp

    return BenchStats(total_time, gpu_time, max_gpu_ram)


#######
#   GROUP
#######


@dataclass
class BenchGroup:
    model: str
    task: str
    input_size: int
    gpu_rams: list[int]
    init_times: list[int]
    total_times: list[int]
    gpu_times: list[int]


def load_group_benches(
        registry, models=MODELS, tasks=TASKS,
        input_size=2000, batch_size=32
):
    for model in models:
        for task in tasks:
            records = query_bench_registry(
                registry,
                input_size=1,
                batch_size=1,
                model=model,
                task=task
            )
            benches = (load_registry_bench(_) for _ in records)
            stats = [bench_stats(_) for _ in benches]

            gpu_rams = [_.max_gpu_ram for _ in stats if _.max_gpu_ram]
            init_times = [_.total_time for _ in stats]

            records = query_bench_registry(
                registry,
                input_size=input_size,
                batch_size=batch_size,
                model=model,
                task=task
            )
            benches = (load_registry_bench(_) for _ in records)
            stats = [bench_stats(_) for _ in benches]

            total_times = [_.total_time for _ in stats]
            gpu_times = [_.gpu_time for _ in stats]

            yield BenchGroup(
                model, task, input_size,
                gpu_rams, init_times, total_times, gpu_times
            )


#######
#
#  SHOW
#
#####


def show_jiant_leaderboard(leaderboard, jiant_scores, models=MODELS, tasks=TASKS):
    leaderboard_index = {
        (model, task): score
        for model, task, score in leaderboard
    }
    jiant_scores_index = {
        (model, task): score
        for model, task, score in jiant_scores
    }

    leaderboard_model_scores = Counter()
    jiant_model_scores = Counter()
    count = len(tasks)
    for model in MODELS:
        for task in tasks:
            value = score_value(leaderboard_index[model, task])
            leaderboard_model_scores[model] += value / count
            value = score_value(jiant_scores_index[model, task])
            jiant_model_scores[model] += value / count

    SCORE = 'score'
    data = []
    for model in models:
        for task in [SCORE] + tasks:
            if task == SCORE:
                leaderboard_score = Score(leaderboard_model_scores[model])
                jiant_score = Score(jiant_model_scores[model])
            else:
                leaderboard_score = leaderboard_index[model, task]
                jiant_score = jiant_scores_index[model, task]
            better = score_value(jiant_score) >= score_value(leaderboard_score)
            parts = [
                '✅' if better else '❌',
                format_score(leaderboard_score, sep='/'),
                format_score(jiant_score, sep='/')
            ]
            value = ' '.join(parts)
            data.append([model, task, value])

    table = pd.DataFrame(data, columns=['model', 'task', 'value'])
    table = table.pivot(index='model', columns='task', values='value')
    table = table.reindex(index=models, columns=[SCORE] + tasks)
    table.columns.name = None
    table.index.name = None

    return table


def show_grid_scores(
        leaderboard, grid_task_scores, task_train_sizes,
        tasks=TASKS, models=MODELS, confs=JIANT_GRID_CONFS,
        cols=3, width=2.5, height=2.5
):
    rows = ceil(len(tasks) / cols)
    fig, axes = plt.subplots(rows, cols)

    id_confs = {_.id: _ for _ in confs}
    task_scores = defaultdict(list)
    for model, task, score in leaderboard:
        score = score_value(score)
        task_scores[task].append(score)

    for ax, task in zip(axes.flatten(), tasks):
        xs, ys, colors = [], [], []

        for x, model in enumerate(models):
            for id, grid_task, score in grid_task_scores:
                score = score_value(score)
                grid_model = id_confs[id].model
                if grid_model == model and grid_task == task:
                    jitter = (random() - 0.5) / 4
                    xs.append(x + jitter)
                    ys.append(score)
                    colors.append('blue')

            for leaderboard_model, leaderboard_task, score in leaderboard:
                score = score_value(score)
                if leaderboard_model == model and leaderboard_task == task:
                    xs.append(x)
                    ys.append(score)
                    colors.append('red')

        ax.scatter(xs, ys, color=colors, s=20, alpha=0.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([])

        title = task
        if task in task_train_sizes:
            train_size = task_train_sizes[task]
            title = f'{task}, {train_size}'
        ax.set_title(title)

        scores = task_scores[task]
        score = statistics.median(scores)
        window = 0.1
        lower, upper = score - window, score + window
        ticks = [lower, score, upper]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{_:0.2f}' for _ in ticks])
        offset = 0.05
        ax.set_ylim(lower - offset, upper + offset)

    for ax in axes[-1]:
        ax.set_xticklabels(models, rotation=90)

    fig.set_size_inches(width * cols, height * rows)
    fig.tight_layout()


def show_seed_scores(leaderboard, grid_task_scores,
                     tasks=TASKS, models=MODELS, confs=JIANT_GRID_CONFS):
    id_confs = {_.id: _ for _ in confs}
    model_task_scores = defaultdict(list)
    for id, task, score in grid_task_scores:
        score = score_value(score)
        model = id_confs[id].model
        model_task_scores[model, task].append(score)

    leaderboard_model_scores = defaultdict(list)
    for model, task, score in leaderboard:
        score = score_value(score)
        leaderboard_model_scores[model].append(score)

    seed(1)
    xs, ys, colors = [], [], []
    samples = 100
    for y, model in enumerate(models):
        for _ in range(samples):
            scores = []
            for task in tasks:
                task_scores = model_task_scores[model, task]
                score = choice(task_scores)
                scores.append(score)
            x = statistics.mean(scores)
            xs.append(x)
            jitter = (random() - 0.5) / 2
            ys.append(y + jitter)
            colors.append('blue')

        scores = leaderboard_model_scores[model]
        x = statistics.mean(scores)
        xs.append(x)
        ys.append(y)
        colors.append('red')

    fig, ax = plt.subplots()
    ax.scatter(xs, ys, s=20, c=colors, alpha=0.4)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    ax.set_xlabel('avg. score over 9 tasks')


def show_seed_scores2(grid_task_scores,
                      tasks=TASKS, models=MODELS, confs=JIANT_GRID_CONFS):
    id_confs = {_.id: _ for _ in confs}
    model_task_scores = defaultdict(list)
    for id, task, score in grid_task_scores:
        score = score_value(score)
        model = id_confs[id].model
        model_task_scores[model, task].append(score)

    seed(1)
    xs, ys, colors = [], [], []
    samples = 100
    for y, model in enumerate(models):
        model_scores = []
        for _ in range(samples):
            sample_scores = []
            for task in tasks:
                task_scores = model_task_scores[model, task]
                score = choice(task_scores)
                sample_scores.append(score)
            score = statistics.mean(sample_scores)
            xs.append(score)
            model_scores.append(score)
            jitter = (random() - 0.5) / 2
            ys.append(y + jitter)
            colors.append('blue')

        xs.extend([
            statistics.mean(model_scores),
            max(model_scores)
        ])
        ys.extend([y + 0.4, y + 0.4])
        colors.extend(['purple', 'red'])

    fig, ax = plt.subplots()
    alpha = 0.4
    ax.scatter(xs, ys, s=20, c=colors, alpha=alpha)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    ax.set_xlabel('avg. score over 9 tasks')

    handles = [
        patches.Patch(color='blue', label='sample', alpha=alpha),
        patches.Patch(color='purple', label='mean', alpha=alpha),
        patches.Patch(color='red', label='max', alpha=alpha),
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.,
    )


def show_gpu_ram_hub_size_bench_report(records, models=MODELS):
    data = []
    for record in records:
        if record.gpu_rams:
            value = statistics.median(record.gpu_rams)
            data.append([record.model, record.task, value])

    table = pd.DataFrame(data, columns=['model', 'task', 'value'])
    table = table.pivot(index='model', columns='task', values='value')
    table = table.mean(axis='columns')

    table = pd.DataFrame({
        'gpu_ram': table,
        'hub_size': MODEL_HUB_SIZES
    })
    table /= GB

    table['ratio'] = table.gpu_ram / table.hub_size
    table = table.applymap('{:0.2f}'.format)
    table = table.reindex(index=models)
    table.index.name = None

    return table


def show_rps_order_bench_report(records):
    data = raw_rps_bench_report_data(records)
    table = bench_report_table(data)

    seed(1)
    samples = []
    for size in range(1, len(TASKS)):
        for _ in range(20):
            columns = sorted(sample(list(table.columns), size))
            if columns not in samples:
                if PARUS not in columns:
                    samples.append(columns)

    # https://material.io/design/color/the-color-system.html
    palette = {
        RUBERT: '#0288D1',
        RUBERT_CONVERSATIONAL: '#7B1FA2',

        BERT_MULTILINGUAL: '#689F38',

        RUGPT3_SMALL: '#FF8A65',
        RUGPT3_MEDIUM: '#F4511E',
        RUGPT3_LARGE: '#BF360C',
    }

    xs, ys, colors = [], [], []
    for y, columns in enumerate(samples):
        slice = table[columns]
        slice = slice.mean(axis='columns')
        min = slice[slice.idxmin()]
        max = slice[slice.idxmax()]
        slice = (slice - min) / (max - min)

        for model, x in slice.items():
            xs.append(x)
            ys.append(y)
            colors.append(palette[model])

    fig, ax = plt.subplots()
    alpha = 0.7
    ax.scatter(xs, ys, c=colors, alpha=alpha)

    labels = [
        ','.join(_)
        for _ in samples
    ]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-1, len(labels))

    ax.set_xticks([])
    ax.set_xlabel('(rps - min) / (max - min)')

    # https://matplotlib.org/3.3.3/tutorials/intermediate/legend_guide.html
    handles = [
        patches.Patch(color=palette[_], label=_, alpha=alpha)
        for _ in MODELS
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.,
    )

    fig.set_size_inches(6, 10)


def show_score_perf(jiant_scores, leaderboard, bench_groups,
                    models=MODELS, tasks=TASKS,
                    width=6, height=6):
    leaderboard_index = {
        (model, task): score
        for model, task, score in leaderboard
    }
    jiant_scores_index = {
        (model, task): score
        for model, task, score in jiant_scores
    }

    data = {}
    for model in models:
        for task in tasks:
            # fill in rucos, muserc
            score = jiant_scores_index.get(
                (model, task),
                leaderboard_index[model, task]
            )
            data[model, task] = score_value(score)

    table = pd.Series(data)
    table = table.unstack()
    model_scores = table.mean(axis='columns')

    gpu_ram_data, rps_data = {}, {}
    for record in bench_groups:
        if record.task in (MUSERC, RUCOS):
            # gpu ram is the same
            # rps scale is diff + missing values
            continue

        if record.task == PARUS:
            # too quick, unstable rps
            continue

        gpu_ram = bench_group_gpu_ram(record) / GB
        rps = bench_group_rps(record)

        gpu_ram_data[record.model, record.task] = gpu_ram
        rps_data[record.model, record.task] = rps

    table = pd.Series(gpu_ram_data)
    table = table.unstack()
    model_gpu_rams = table.mean(axis='columns')

    table = pd.Series(rps_data)
    table = table.unstack()
    model_rpses = table.mean(axis='columns')

    table = pd.DataFrame({
        'score': model_scores,
        'gpu_ram': model_gpu_rams,
        'rps': model_rpses
    })

    sizes, xs, ys, texts = [], [], [], []
    for record in table.itertuples():
        x, y = record.rps, record.score
        size = record.gpu_ram * 200
        model = record.Index

        sizes.append(size)
        xs.append(x)
        ys.append(y)

        text = f'{model}, {record.gpu_ram:0.1f}gb'
        texts.append(text)

    fig, ax = plt.subplots()
    ax.scatter(
        xs, ys,
        s=sizes,
        alpha=0.5
    )
    for text, x, y, size in zip(texts, xs, ys, sizes):
        offset = sqrt(size) / 1.7
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(offset, 0),
            textcoords='offset points'
        )

    ax.set_ylabel('avg. score over 9 tasks')
    ax.set_xlabel('inference speed, records per second')

    min, max = ax.get_xlim()
    offset = (max - min) / 25
    ax.set_xlim(min - offset, max + offset)

    min, max = ax.get_ylim()
    offset = (max - min) / 25
    ax.set_ylim(min - offset, max + offset)

    fig.set_size_inches(width, height)


#########
#  BENCH
######


def show_benches(benches, width=8, height=6):
    tables = []
    for bench in benches:
        table = pd.DataFrame(bench)
        tables.append(table)

    keys = ['cpu_usage', 'ram', 'gpu_usage', 'gpu_ram']
    lims = {
        key: max(table[key].max() for table in tables)
        for key in keys
    }

    fig, axes = plt.subplots(len(keys), 1, sharex=True)
    axes = axes.flatten()

    for key, ax in zip(keys, axes):
        for table in tables:
            x = table.timestamp - table.timestamp.min()
            ax.plot(x, table[key], color='blue', alpha=0.3)

        lim = lims[key]
        ax.set_yticks([lim])
        if key in ['ram', 'gpu_ram']:
            label = '{} mb'.format(int(lim / MB))
        else:
            label = '{}%'.format(int(lim * 100))
        ax.set_yticklabels([label])
        ax.set_ylabel(key)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('right')

    fig.set_size_inches(width, height)
    fig.tight_layout()


def show_bench(bench, **kwargs):
    show_benches([bench], **kwargs)


def bench_report_table(data, models=MODELS, tasks=TASKS):
    table = pd.DataFrame(data, columns=['model', 'task', 'value'])
    table = table.pivot(index='model', columns='task', values='value')
    table.columns.name = None
    table.index.name = None
    return table.reindex(index=models, columns=tasks)


def gpu_ram_bench_report_data(records):
    for record in records:
        if record.gpu_rams:
            value = ' '.join(
                '{:0.1f}'.format(_ / GB)
                for _ in record.gpu_rams
            )
        else:
            value = ''
        yield record.model, record.task, value


def show_gpu_ram_bench_report(records):
    data = gpu_ram_bench_report_data(records)
    return bench_report_table(data)


def bench_group_gpu_ram(record):
    if record.gpu_rams:
        return statistics.median(record.gpu_rams)


def gpu_ram_bench_report_data2(records):
    for record in records:
        gpu_ram = bench_group_gpu_ram(record)
        if gpu_ram is None:
            value = ''
        else:
            value = '{:0.2f}'.format(gpu_ram / GB)
        yield record.model, record.task, value


def show_gpu_ram_bench_report2(records):
    data = gpu_ram_bench_report_data2(records)
    return bench_report_table(data)


def init_time_bench_report_data(records):
    for record in records:
        if record.init_times:
            value = ' '.join(
                '{:0.0f}'.format(_)
                for _ in sorted(record.init_times)
            )
        else:
            value = ''
        yield record.model, record.task, value


def show_init_time_bench_report(records):
    data = init_time_bench_report_data(records)
    return bench_report_table(data)


def proc_time_bench_report_data(records):
    for record in records:
        if record.init_times and record.total_times:
            init_time = statistics.median(record.init_times)
            value = ' '.join(
                '{:0.0f}'.format(_ - init_time)
                for _ in sorted(record.total_times)
            )
        else:
            value = ''
        yield record.model, record.task, value


def show_proc_time_bench_report(records):
    data = proc_time_bench_report_data(records)
    return bench_report_table(data)


def bench_group_rps(record, input_size=2000):
    if not record.init_times or not record.total_times:
        return

    init_time = statistics.median(record.init_times)
    proc_time = statistics.median(
        _ - init_time
        for _ in record.total_times
    )
    return input_size / proc_time


def rps_bench_report_data(records, input_size=2000):
    for record in records:
        rps = bench_group_rps(record, input_size)
        if rps is None:
            value = ''
        else:
            value = '{:0.0f}'.format(rps)
        yield record.model, record.task, value


def show_rps_bench_report(records):
    data = rps_bench_report_data(records)
    return bench_report_table(data)


def raw_rps_bench_report_data(records, input_size=2000):
    for record in records:
        value = bench_group_rps(record, input_size)
        yield record.model, record.task, value
