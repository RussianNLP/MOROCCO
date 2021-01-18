
import re
import sys
from datetime import datetime
from os import (
    environ,
    makedirs,
    listdir,
    rmdir,
    remove,
    rename
)
from os.path import (
    join,
    exists,
    dirname,
    expanduser,
    isdir
)
from shutil import (
    copy,
    copytree
)
from shutil import rmtree
from fnmatch import fnmatch
import logging
import json
import subprocess
from io import StringIO
from contextlib import contextmanager
from importlib import reload
from tempfile import (
    NamedTemporaryFile,
    TemporaryDirectory
)
from collections import (
    Counter,
    defaultdict,
    namedtuple
)
import argparse

import torch
import pytorch_pretrained_bert
import transformers
import allennlp

from tqdm import tqdm as log_progress

from jiant.utils.config import (
    params_from_file,
    write_params
)
from jiant.utils.options import parse_cuda_list_arg
from jiant.preprocess import build_tasks
from jiant.models import build_model
from jiant.utils.utils import load_model_state
from jiant.tasks import REGISTRY
from jiant import evaluate
from jiant.__main__ import main as jiant_main


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

HUMAN = 'human'
TFIDF = 'tfidf'

RUBERT = 'rubert'  # 700MB
RUBERT_CONVERSATIONAL = 'rubert-conversational'
BERT_MULTILINGUAL = 'bert-multilingual'

RUGPT3_LARGE = 'rugpt3-large'  # 3GB
RUGPT3_MEDIUM = 'rugpt3-medium'  # 1.7GB
RUGPT3_SMALL = 'rugpt3-small'  # 700MB

MODELS = [
    RUBERT,
    RUBERT_CONVERSATIONAL,
    BERT_MULTILINGUAL,

    RUGPT3_LARGE,
    RUGPT3_MEDIUM,
    RUGPT3_SMALL,
]
MODEL_HUB_NAMES = {
    RUBERT_CONVERSATIONAL: 'DeepPavlov/rubert-base-cased-conversational',
    RUBERT: 'DeepPavlov/rubert-base-cased',
    BERT_MULTILINGUAL: 'bert-base-multilingual-cased',

    RUGPT3_LARGE: 'sberbank-ai/rugpt3large_based_on_gpt2',
    RUGPT3_MEDIUM: 'sberbank-ai/rugpt3medium_based_on_gpt2',
    RUGPT3_SMALL: 'sberbank-ai/rugpt3small_based_on_gpt2',
}

DATA_DIR = expanduser('~/data')

PRIVATE = 'private'
PUBLIC = 'public'

TEST = 'test'
VAL = 'val'

EXPS_DIR = expanduser('~/exps')
GRID_PREDS_DIR = expanduser('~/preds')

JIANT_DIR = expanduser('~/jiant-v1-legacy')
JIANT_CONF = join(JIANT_DIR, 'jiant/config/superglue_bert.conf')

# access to single bucket
# bucket is capped by size
S3_KEY_ID = '5lcyb03uDlKWQ9E-5Cie'
S3_KEY = 'AzMOIwDdIdDGxKU7ZCpVr6_8kx_7x_yzzGHFlIeS'
S3_BUCKET = 'russian-superglue'
S3_REGION = 'us-east-1'
S3_ENDPOINT = 'https://storage.yandexcloud.net'


#########
#
#   CONTEXT
#
#######


@contextmanager
def env(**vars):
    original = dict(environ)
    environ.update(vars)
    try:
        yield
    finally:
        environ.clear()
        environ.update(original)


LOGGER = logging.getLogger()


@contextmanager
def no_loggers(loggers):
    for logger in loggers:
        logger.disabled = True
    try:
        yield
    finally:
        for logger in loggers:
            logger.disabled = False


######
#
#   LOG
#
#####


def log(format, *args):
    message = format % args
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        '[%s] %s' % (now, message),
        file=sys.stderr,
        flush=True
    )


#####
#
#  IO
#
#####


def load_lines(path):
    with open(path) as file:
        for line in file:
            yield line.rstrip('\n')


def dump_lines(lines, path):
    with open(path, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def parse_jl(lines):
    for line in lines:
        yield json.loads(line)


def format_jl(items):
    for item in items:
        yield json.dumps(item, ensure_ascii=False)


def load_jl(path):
    lines = load_lines(path)
    return parse_jl(lines)


def dump_jl(items, path):
    lines = format_jl(items)
    dump_lines(lines, path)


def parse_tsv(lines, sep='\t'):
    for line in lines:
        yield line.split(sep)


#####
#
#  PATH
#
#####


def maybe_mkdir(dir):
    makedirs(dir, exist_ok=True)


def maybe_copytree(source, target):
    if not exists(target):
        copytree(source, target)


def maybe_rmdir(path):
    if exists(path):
       rmtree(path) 


def rm_any(path):
    if isdir(path):
        rmtree(path)
    else:
        remove(path)


#####
#
#  S3
#
#####


def patch_s3_args(args, bucket=S3_BUCKET):
    for arg in args:
        yield re.sub('^//', f's3://{bucket}/', arg)


def s3_call(args, key_id=S3_KEY_ID, key=S3_KEY,
            region=S3_REGION, endpoint=S3_ENDPOINT):
    with env(
        AWS_ACCESS_KEY_ID=key_id,
        AWS_SECRET_ACCESS_KEY=key
    ):
        command = ['aws', '--region', region, '--endpoint-url', endpoint, 's3']
        args = list(patch_s3_args(args))
        log(f'Call S3: {args!r}')
        subprocess.run(command + args)


#######
#
#   TRAIN
#
######


def train_jiant(model, task, exps_dir, data_dir, config=JIANT_CONF, seed=1):
    target_tasks = task
    if task == TERRA:
        target_tasks = f'"{TERRA},{LIDIRUS}"'

    input_module = MODEL_HUB_NAMES[model]

    task_specs = {
        DANETQA: 'batch_size = 4, val_interval = 1000',
        RCB: 'batch_size = 4, val_interval = 60',
        PARUS: 'batch_size = 4, val_interval = 100',
        MUSERC: 'batch_size = 4, val_interval = 1000, val_data_limit = -1',
        RUCOS: 'batch_size = 8, val_interval = 10000, val_data_limit = -1',
        TERRA: 'batch_size = 4, val_interval = 625',
        RUSSE: 'batch_size = 4, val_interval = 1000',
        RWSD: 'batch_size = 4, val_interval = 139, optimizer = adam'
    }
    spec = task_specs[task]

    with env(
        JIANT_PROJECT_PREFIX=exps_dir,
        JIANT_DATA_DIR=data_dir,
        WORD_EMBS_FILE='None'
    ):
        jiant_main([
            '--config_file', config,
            '--overrides',
            f'input_module = {input_module}, exp_name = {model}, '
            f'random_seed = {seed}, cuda = 0, run_name = {task}, '
            f'pretrain_tasks = {task}, target_tasks = {target_tasks}, do_pretrain = 1, '
            'do_target_task_training = 0, do_full_eval = 1, '
            f'batch_size = 4, val_interval = 100, {spec}'
        ])


######
#
#   EXPS
#
######


def patch_exp_params(params, model):
    for key in ['data_dir', 'exp_dir', 'local_log_path', 'project_dir', 'run_dir']:
        params[key] = None

    params.pool_type = 'max'  # by default auto, whyyy?
    params.tokenizer = MODEL_HUB_NAMES[model]  # by default tokenizer=auto, why?


def is_best_model(filename):
    return fnmatch(filename, 'model_*.best.th')


def strip_exp(exps_dir, model, task):
    exp_dir = join(exps_dir, model)
    
    for subdir in ['preproc', 'tasks']:
        dir = join(exp_dir, subdir)
        for item in listdir(dir):
            # rwsd__test_data
            # rwsd__train_data
            # rwsd.DeepPavlov
            name = re.match('([^_\.]+)', item).group(1)
            if name == task or (task == TERRA and name == LIDIRUS):
                rm_any(join(dir, item))
        
        if not listdir(dir):
            rmdir(dir)

    dir = join(exp_dir, task)
    for item in listdir(dir):
        # metric_state_pretrain_val_10.th
        # metric_state_pretrain_val_3.best.th
        # model_state_pretrain_val_10.th
        # params.conf
        # RWSD.jsonl
        # tensorboard
        # training_state_pretrain_val_3.best.th
        if is_best_model(item):
            rename(
                join(dir, item),
                join(dir, 'model.th')
            )
        elif item not in ('log.log', 'params.conf'):
            rm_any(join(dir, item))

    path = join(dir, 'params.conf')
    with no_loggers([LOGGER]):
        params = params_from_file(path)
    patch_exp_params(params, model)
    write_params(params, path)


#######
#
#   INFER
#
#######


RWSD_ITEM = {'idx': 0, 'target': {'span1_text': 'Члены городского совета', 'span2_text': 'они опасались', 'span1_index': 0, 'span2_index': 10}, 'label': True, 'text': 'Члены городского совета отказали организаторам митинга в разрешении, потому что они опасались насилия.'}


def dump_task(data_dir, task, items):
    title = TASK_TITLES[task]
    dir = join(data_dir, title)
    maybe_mkdir(dir)

    if task == LIDIRUS:
        path = join(dir, title + '.jsonl')
        dump_jl(items, path)
        
    else:
        path = join(dir, 'test.jsonl')
        dump_jl(items, path)

        for filename in ['train.jsonl', 'val.jsonl']:
            data = []
            if task == RWSD:
                # rwsd load_data breaks on empty train
                data = [RWSD_ITEM]

            path = join(dir, filename)
            dump_jl(data, path)

    return dir


def load_preds(dir, task):
    path = join(dir, TASK_TITLES[task] + '.jsonl')
    return load_jl(path)


def infer_jiant(exp_dir, task, items, batch_size=4):
    # use cached tokenizer
    path = join(exp_dir, 'transformers_cache')
    with env(PYTORCH_TRANSFORMERS_CACHE=path):
        reload(transformers.file_utils)

    # use terra model for lidirus        
    run_dir = join(
        exp_dir,
        TERRA if task == LIDIRUS else task
    )

    loggers = [
        LOGGER,
        pytorch_pretrained_bert.modeling.logger,
        transformers.file_utils.logger,
        transformers.configuration_utils.logger,
        transformers.modeling_utils.logger,
        transformers.tokenization_utils.logger,
        allennlp.nn.initializers.logger
    ]
    with no_loggers(loggers):
        path = join(run_dir, 'params.conf')
        args = params_from_file(path)
        cuda_device = parse_cuda_list_arg('auto')    

    args.local_log_path = join(run_dir, 'log.log')
    args.exp_dir = args.project_dir = exp_dir
    args.run_dir = run_dir

    log('Build tasks')
    with no_loggers(loggers), TemporaryDirectory() as dir:
        args.exp_dir = args.data_dir = dir  # hide pkl, preproc
        dump_task(dir, task, items=[])  # mock empty train, val, test
        if task in (TERRA, LIDIRUS):
            dump_task(dir, LIDIRUS if task == TERRA else TERRA, items=[])
        _, tasks, vocab, word_embs = build_tasks(args, cuda_device)

    log('Build model, load transformers pretrain')
    with no_loggers(loggers):
        args.exp_dir = exp_dir  # use transformers cache
        model = build_model(args, vocab, word_embs, tasks, cuda_device)

    path = join(run_dir, 'model.th')
    log(f'Load state {path!r}')
    load_model_state(model, path, cuda_device)

    log(f'Build mock task, infer via eval, batch_size={batch_size}')
    with no_loggers(loggers), TemporaryDirectory() as dir:
        args.exp_dir = args.data_dir = dir
        dump_task(dir, task, items)

        if task in (TERRA, LIDIRUS):
            # choose one at inference
            args.pretrain_tasks = task
            args.target_tasks = task

        _, tasks, _, _ = build_tasks(args, cuda_device)
        _, preds = evaluate.evaluate(
            model, tasks,
            batch_size, cuda_device, 'test'
        )
        evaluate.write_preds(
            tasks, preds, dir,
            'test', args.write_strict_glue_format
        )

        return list(load_preds(dir, task))


#####
#
#  TASK
#
######


def load_task(task, access, split, dir=DATA_DIR):
    title = TASK_TITLES[task]

    name = split
    if split == TEST:
        if access == PUBLIC and task == LIDIRUS:
            name = title
        elif access == PRIVATE:
            name = 'test_with_answers'

    path = join(dir, access, title, f'{name}.jsonl')
    return load_jl(path)


######
#
#  EVAL
#
######


# danetqa PairClassificationTask 2
# lidirus GLUEDiagnosticTask
# muserc MuSeRCTask
# parus MultipleChoiceTask 2
# rcb PairClassificationTask 3
# rucos RuCoSTask
# russe PairClassificationTask 2
# rwsd SpanClassificationTask 2
# terra RTETask PairClassificationTask 2

# PairClassificationTask
#   logits = out["logits"]
#   labels = out["labels"]

# GLUEDiagnosticTask
#   out["logits"], batch["labels"]

# MultipleChoiceTask
#   logits = out["logits"]
#   labels = batch["label"]

# SpanClassificationTask
#   logits = out["logits"]
#   labels = batch["labels"]

# MuSeRCTask
#   logits, labels = out["logits"], batch["label"]
#   idxs = [(p, q) for p, q in zip(batch["psg_idx"], batch["qst_idx"])]

# RuCoSTask
#   logits = out["logits"]
#   anss = batch["ans_str"]
#   idxs = [(p, q) for p, q in zip(batch["psg_idx"], batch["qst_idx"])]


def eval_task(name):
    Task, _, _ = REGISTRY[name]
    return Task(
        name=name,
        path='/',  # fake for os.path.join
        tokenizer_name='',  # space tokenizer
        max_seq_len=10,
    )


def eval_muserc(preds, targets):
    labels, ids, passage_ids, question_ids = [], [], [], []
    for passage in preds:
        for question in passage['passage']['questions']:
            for answer in question['answers']:
                ids.append(answer['idx'])
                labels.append(answer['label'])
                passage_ids.append(passage['idx'])
                question_ids.append(question['idx'])
    logits = torch.nn.functional.one_hot(
        torch.tensor(labels),
        num_classes=2
    )

    id_labels = {}
    for passage in targets:
        for question in passage['passage']['questions']:
            for answer in question['answers']:
                id_labels[answer['idx']] = answer['label']
    labels = [id_labels[_] for _ in ids]
    labels = torch.tensor(labels)

    out = {'logits': logits}
    batch = {
        'label': labels,
        'psg_idx': passage_ids,
        'qst_idx': question_ids
    }
    task = eval_task(MUSERC)
    task.update_metrics(out, batch)
    return task.get_metrics()


def strip_rucos_label(label):
    # whyyy? 'России\n', 'ГДР ', 'УПЦ КП,'
    return re.sub(r'\W+$', '', label)


def eval_rucos(preds, targets):
    id_preds = {
        _['idx']: strip_rucos_label(_['label'])
        for _ in preds
    }

    id_answers = {}
    logits, strings, passage_ids, question_ids = [], [], [], []
    for passage in targets:
        passage_id = 'test-%d' % passage['idx']
        text = passage['passage']['text']
        entities = [
            text[_['start']:_['end']]
            for _ in passage['passage']['entities']
        ]
        for question in passage['qas']:
            question_id = question['idx']
            answers = [_['text'] for _ in question['answers']]
            id_answers[passage_id, question_id] = answers
            pred = id_preds.get(question_id)
            if not pred:
                continue
            for entity in entities:
                logit = [0., 1.] if pred == entity else [1., 0.]
                logits.append(logit)
                strings.append(entity)
                passage_ids.append(passage_id)
                question_ids.append(question_id)

    out = {'logits': torch.tensor(logits)}
    batch = {
        'ans_str': strings,
        'psg_idx': passage_ids,
        'qst_idx': question_ids
    }

    task = eval_task(RUCOS)
    task._answers = id_answers

    task.update_metrics(out, batch)
    return task.get_metrics()


def lidirus_masks(item):
    selected = {
        'logic': 'logic',
        'predicate-argument-structure': 'pr_ar_str',
        'lexical-semantics': 'lex_sem',
        'knowledge': 'knowledge'
    }
    for key in item.keys() & selected.keys():
        value = item[key]
        key = selected[key]
        yield key
        yield '%s__%s' % (key, value)


def eval_lidirus(preds, targets):
    renames = {
        'not_entailment': 0,
        'entailment': 1,
    }
    ids, labels = [], []
    for item in preds:
        id = item['idx']
        label = renames[item['label']]
        ids.append(id)
        labels.append(label)
    logits = torch.nn.functional.one_hot(
        torch.tensor(labels),
        num_classes=2
    )

    id_targets, tag_masks = {}, {}
    for item in targets:
        id = int(item['idx'])  # whyy str?
        id_targets[id] = item
        for tag in lidirus_masks(item):
            tag_masks[tag] = []
    
    labels = []
    for id in ids:
        item = id_targets[id]
        label = renames[item['label']]
        labels.append(label)
        tags = set(lidirus_masks(item))
        for tag in tag_masks:
            tag_masks[tag].append(tag in tags)

    out = {'logits': logits}
    batch = {'labels': torch.tensor(labels)}
    for tag, mask in tag_masks.items():
        batch[tag] = torch.tensor(mask)

    task = eval_task(LIDIRUS)
    with TemporaryDirectory() as dir:
        path = dump_task(dir, LIDIRUS, targets)
        task.path = path
        with no_loggers([LOGGER]):
            task.load_data()

    task.update_metrics(out, batch)
    return task.get_metrics()


def eval_other(name, preds, targets):
    renames = {
        'false': 0,
        'False': 0,
        False: 0,
        'true': 1,
        'True': 1,
        True: 1,
        'neutral': 0,
        'not_entailment': 0,
        'entailment': 1,
        'contradiction': 2,

    }
    labels = [_['label'] for _ in preds]
    labels = [renames.get(_, _) for _ in labels]
    logits = torch.nn.functional.one_hot(
        torch.tensor(labels),
        num_classes=(3 if name == RCB else 2)
    )

    id_labels = {}
    for item in targets:
        id, label = item['idx'], item['label']
        label = renames.get(label, label)
        id_labels[id] = label
    labels = [id_labels[_['idx']] for _ in preds]
    labels = torch.tensor(labels)

    out = {
        'logits': logits,
        'labels': labels
    }
    batch = {
        'label': labels,
        'labels': labels
    }
    task = eval_task(name)
    task.update_metrics(out, batch)
    return task.get_metrics()


def eval(name, preds, targets):
    if name == MUSERC:
        return eval_muserc(preds, targets)
    elif name == RUCOS:
        return eval_rucos(preds, targets)
    elif name == LIDIRUS:
        return eval_lidirus(preds, targets)
    else:
        return eval_other(name, preds, targets)


TASK_METRICS = {
    DANETQA: 'accuracy',
    MUSERC: 'ans_f1',
    PARUS: 'accuracy',
    RCB: 'f1',
    RUCOS: 'f1',
    RUSSE: 'accuracy',
    RWSD: 'acc',
    TERRA: 'accuracy',
    LIDIRUS: 'all_mcc',
}


def select_score(task, metrics):
    return metrics[TASK_METRICS[task]]


######
#
#  GRID
#
######


GridConf = namedtuple(
    'GridConf',
    ['id', 'model', 'seed']
)


GRID_CONFS = [
    GridConf('01', RUBERT_CONVERSATIONAL, seed=111),
    GridConf('02', RUBERT_CONVERSATIONAL, seed=2),
    GridConf('03', RUBERT_CONVERSATIONAL, seed=3),

    GridConf('04', RUBERT, seed=1),
    GridConf('05', RUBERT, seed=2),
    GridConf('06', RUBERT, seed=3),
    GridConf('07', RUBERT, seed=4),
    GridConf('08', RUBERT, seed=5),
    GridConf('09', RUBERT, seed=6),
    GridConf('10', RUBERT, seed=7),
    GridConf('11', RUBERT, seed=8),

    GridConf('12', RUBERT_CONVERSATIONAL, seed=4),
    GridConf('13', RUBERT_CONVERSATIONAL, seed=5),
    GridConf('14', RUBERT_CONVERSATIONAL, seed=6),
    GridConf('15', RUBERT_CONVERSATIONAL, seed=7),
    GridConf('16', RUBERT_CONVERSATIONAL, seed=8),

    GridConf('17', RUBERT, seed=9),

    GridConf('18', RUGPT3_SMALL, seed=1),
    GridConf('19', RUGPT3_MEDIUM, seed=1),
    GridConf('20', BERT_MULTILINGUAL, seed=1),
]


def find_grid_exp_dir(exps_dir, conf_id, model, task):
    dir = join(exps_dir, conf_id)
    if exists(dir):
        model_dir = join(dir, model)
        if exists(model_dir):
            dir = model_dir
        if task == LIDIRUS:
            task = TERRA
        if exists(join(dir, task)):
            return dir


def grid_exp_finised(exp_dir, task):
    if task == LIDIRUS:
        task = TERRA
    return exists(join(exp_dir, task, 'model.th'))


def dump_grid_preds(conf_id, task, items, dir=GRID_PREDS_DIR):
    dir = join(dir, conf_id)
    maybe_mkdir(dir)
    path = join(dir, f'{task}.jl')
    dump_jl(items, path)


def grid_preds_exist(conf_id, task, dir=GRID_PREDS_DIR):
    return exists(join(dir, conf_id, f'{task}.jl'))


def load_grid_preds(conf_id, task, dir=GRID_PREDS_DIR):
    path = join(dir, conf_id, f'{task}.jl')
    return load_jl(path)


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
        # yep, drop second
        value, _ = value.split(sep)
    return float(value)


def parse_leaderboard(records, name_offset=1, scores_offset=5):
    header = next(records)
    tasks = [LEADERBOARD_RENAMES[_] for _ in header[scores_offset:]]
    for record in records:
        model = LEADERBOARD_RENAMES[record[name_offset]]
        scores = [parse_leaderboard_score(_) for _ in record[scores_offset:]]
        yield model, dict(zip(tasks, scores))


#######
#
#   CLI
#
######


def cli_serve(args):
    log('Reading items from stdin')
    items = list(parse_jl(sys.stdin))
    log(f'Read {len(items)} items')

    preds = infer_jiant(
        args.exp_dir, args.task, items,
        batch_size=args.batch_size
    )
    
    log('Writing preds to stdout')
    lines = format_jl(preds)
    for line in lines:
        print(line)


def cli_train(args):
    train_jiant(
        args.model, args.task,
        args.exps_dir, args.data_dir,
        seed=args.seed
    )
    strip_exp(args.exps_dir, args.model, args.task)


def cli_s3(args):
    s3_call(args.args)


def existing_path(path):
    if not exists(path):
        raise argparse.ArgumentTypeError(f'{path!r} does not exist')
    return path


def main(args):
    parser = argparse.ArgumentParser(prog='main.py')
    parser.set_defaults(function=None)
    subs = parser.add_subparsers()

    sub = subs.add_parser('serve')
    sub.set_defaults(function=cli_serve)
    sub.add_argument('exp_dir', type=existing_path)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('--batch-size', type=int, default=128)

    sub = subs.add_parser('train')
    sub.set_defaults(function=cli_train)
    sub.add_argument('model', choices=MODELS)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('exps_dir')
    sub.add_argument('data_dir', type=existing_path)
    sub.add_argument('--seed', type=int, default=1)

    sub = subs.add_parser('s3')
    sub.set_defaults(function=cli_s3)
    sub.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args(args)
    if not args.function:
        parser.print_help()
        parser.exit()
    try:
        args.function(args)
    except (KeyboardInterrupt, BrokenPipeError):
        pass


if __name__ == '__main__':
    main(sys.argv[1:])
    
