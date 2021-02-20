
import re
import sys
from datetime import datetime
from time import (
    time,
    sleep
)
from os import (
    stat,
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
    copytree,
    rmtree
)
from fnmatch import fnmatch
from uuid import uuid1
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
    defaultdict
)
from itertools import (
    cycle,
    islice
)
from math import (
    ceil,
    sqrt
)
from random import (
    seed,
    random,
    sample,
    choice
)
import statistics
import argparse

from tqdm import tqdm as log_progress

import pandas as pd

from matplotlib import patches
from matplotlib import pyplot as plt

try:
    import torch
    import pytorch_pretrained_bert
    import transformers
    import allennlp

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
except ImportError:
    # just analyse benches locally
    pass


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

PROJ_DIR = expanduser('~')
PROJ_DIR = '..'

DATA_DIR = join(PROJ_DIR, 'data')

PRIVATE = 'private'
PUBLIC = 'public'

TRAIN = 'train'
TEST = 'test'
VAL = 'val'

EXPS_DIR = join(PROJ_DIR, 'exps')
GRID_PREDS_DIR = join(PROJ_DIR, 'preds/grid')
CONF_TASK_METRICS = join(GRID_PREDS_DIR, 'conf_task_metrics.jl')
BENCHES_DIR = join(PROJ_DIR, 'benches')

JIANT_DIR = join(PROJ_DIR, 'jiant-v1-legacy')
JIANT_CONF = join(JIANT_DIR, 'jiant/config/superglue_bert.conf')

# bucket is capped by size
S3_KEY_ID = '5lcyb03uDlKWQ9E-5Cie'
S3_KEY = 'AzMOIwDdIdDGxKU7ZCpVr6_8kx_7x_yzzGHFlIeS'
S3_BUCKET = 'russian-superglue'
S3_REGION = 'us-east-1'
S3_ENDPOINT = 'https://storage.yandexcloud.net'

# service account keys, access to s3 and containter registry, can push and pull
# yc iam key create --service-account-name russian-superglue --folder-name russian-superglue -o key.json
DOCKER_KEY = r'''
{
   "id": "aje6lss13v325arf8bbt",
   "service_account_id": "ajetj9nn233tvqpsa5jp",
   "created_at": "2021-01-18T12:25:27.732988Z",
   "key_algorithm": "RSA_2048",
   "public_key": "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAh5UL26TN9CXbEGN8PI4y\nIOPeFiU+ItkExRI0ohmE5O8BBzqsaqjMADPrs46q9hBz/zxcb9J+4YZUErr0Eayl\nLsh5YJG8Tjt4v3JLskwlGV3r00QkSF/WzgnMig9e+MtMMipQxxNY5L/lwQ6zUapr\nUE2fUDoO/Q+AXsh8+K0qAGrZ8XfGal4FaimnOx0eZWV9d79Hjf6blQs+CAoRo39u\nyy3+FIOmmGSSQom0V19KawzrCqHESm+c9HXJDjtDatlkFGJpihncx8aHbaB5yP8e\ntFfT94yy2HJGlDcamy8lwIfF55CepWH9VKi24jPzvlB+laEav8AWMl0F80L2Ei4F\ntwIDAQAB\n-----END PUBLIC KEY-----\n",
   "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCHlQvbpM30JdsQ\nY3w8jjIg494WJT4i2QTFEjSiGYTk7wEHOqxqqMwAM+uzjqr2EHP/PFxv0n7hhlQS\nuvQRrKUuyHlgkbxOO3i/ckuyTCUZXevTRCRIX9bOCcyKD174y0wyKlDHE1jkv+XB\nDrNRqmtQTZ9QOg79D4BeyHz4rSoAatnxd8ZqXgVqKac7HR5lZX13v0eN/puVCz4I\nChGjf27LLf4Ug6aYZJJCibRXX0prDOsKocRKb5z0dckOO0Nq2WQUYmmKGdzHxodt\noHnI/x60V9P3jLLYckaUNxqbLyXAh8XnkJ6lYf1UqLbiM/O+UH6VoRq/wBYyXQXz\nQvYSLgW3AgMBAAECggEAcOd2U3cfJtQrY69k8sx7BBadh5WF8+EC7tVlTSOxHi/F\njG6Yy8067GRQQFtrFLPI1OBAHcKNdGK3Ok3DF8DMYXQCV5+kzwUJXWvhi12Le19S\nFcFl8XsV1sgkM0fvj4FEN3YzhEQhP4Rp4GXMgBJJgTrSky1189hnzwZnw3H4SxVy\nh0K2hP6f+aI+DFxCx8x8lzSkNoh0ANN/VUfd8FzvxTf/QqOxI8xrI3n8cADHZK1T\n5g1aOj5+xzpLn82XhA/RHeDH3GC5MT6E/w947evjlz2ejLfZZbGkA4q3uNJ9Ps2+\nhfw1N84gMtAfvv9DICyLOa4t6WadoC8/Rq7BNrIG4QKBgQDmfB8wBfUeEOmwCnTG\n0A+IVXquevB+NsIWYm+dZG7/KjYf554+M4hQPyezC+aT4QODrRyr67NPXruV+e0O\nXxNuu/eBRVRWT0NVhuV4VL/msSj/cO8F4U24JYP6RuIaNWrLtrjMBlY3z2hat8BM\nsdwcPD5idVWWmaxWyMqa6XPcZwKBgQCWl2jRkpg6sFCMjSwx6BgbjYplHRvsAXsU\nBr54Dd9Ypdo5eVZmj/FURmPlVnYHLmFE3N1GrfDeNAvCT+ypIqYRzRjL097v8+e2\neXTO2kgrzm8Zhpv+7NF0pqXm1h9RpVu0q3nH4RmgzlPnaCBTINoLvMDjs4UmwPdG\nGN/1GXK6MQKBgQCigBSFCU4+enIoWdbnbT3mQ04Rwj/Y3MaOfaxv7aWMZbhvWh/p\nfu+2lDBrPhK9ZEHUDrNOSwnLBeF/5gvKvCG5SvE/xR+nEo9It6kF48rA3Vsobfk3\nzhe7o0efp2Y2UD8RjaxQvI8BHkxW2YLNEAE+LwNU66ECYypsrXibK8kyNQKBgBeG\nY6uJmRph/NNYInVRaqKzQ9Fz8K63tIB2ZT7f++ofTq331JWFGxAtRuHG1f1dM3jM\ngAzQk3ZC7ytVzQTHEuZpAdylpogZtDL/Wk4OL4QYZaa5LpluaXItrnEXNiFNEbxx\npT6iXZyPXvAhhhs2YJnAzOlFXCGnt3lN3X6ukQuhAoGBALyZ1KyqiGLWyYtw5A9N\n+hPgpT5NP5o0UkfP/9cHx8tgVisgGqhC0e4EhvJVaO2b0VeIh5qHWtmLtFCC+RNH\n5nJ3mD9eZJR3P5zy6y7Wg1hE21nKh4WdhOlp/Wo/MJcsInbAs5VFqklkeW0hmb8E\n7khYecg0qRJkOLLKbWZ2PbM+\n-----END PRIVATE KEY-----\n"
}
'''
DOCKER_REGISTRY = 'cr.yandex/crpdsbu4ons2b57kp60d'


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


def write_lines(lines, file):
    for line in lines:
        file.write(line + '\n')


def dump_lines(lines, path):
    with open(path, 'w') as file:
        write_lines(lines, file)


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


def load_text(path):
    with open(path) as file:
        return file.read()


def dump_text(text, path):
    with open(path, 'w') as file:
        file.write(text)


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


def path_modified(path):
    record = stat(path)
    return datetime.fromtimestamp(record.st_mtime)


####
#
#  RECORD
#
#####


class Record(object):
    __attributes__ = []

    def __init__(self, *args, **kwargs):
        for key in self.__attributes__:
            setattr(self, key, None)

        for key, value in zip(self.__attributes__, args):
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and all(
                (getattr(self, _) == getattr(other, _))
                for _ in self.__attributes__
            )
        )

    def __ne__(self, other):
        return not self == other

    def __iter__(self):
        return (getattr(self, _) for _ in self.__attributes__)

    def __hash__(self):
        return hash(tuple(self))

    @property
    def as_json(self):
        data = {}
        for key in self.__attributes__:
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_json(cls, data):
        args = []
        for key in cls.__attributes__:
            value = data.get(key)
            args.append(value)
        return cls(*args)

    def __repr__(self):
        name = self.__class__.__name__
        args = ', '.join(
            '{key}={value!r}'.format(
                key=_,
                value=getattr(self, _)
            )
            for _ in self.__attributes__
        )
        return '{name}({args})'.format(
            name=name,
            args=args
        )

    def _repr_pretty_(self, printer, cycle):
        name = self.__class__.__name__
        if cycle:
            printer.text('{name}(...)'.format(name=name))
        else:
            printer.text('{name}('.format(name=name))
            keys = self.__attributes__
            size = len(keys)
            if size:
                with printer.indent(4):
                    printer.break_()
                    for index, key in enumerate(keys):
                        printer.text(key + '=')
                        value = getattr(self, key)
                        printer.pretty(value)
                        if index < size - 1:
                            printer.text(',')
                            printer.break_()
                printer.break_()
            printer.text(')')


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


#######
#
#  SCORE
#
#####


class Score(Record):
    __attributes__ = ['first', 'second']


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


def select_score(task, metrics):
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


######
#
#  GRID
#
######


class GridConf(Record):
    __attributes__ = ['id', 'model', 'seed']


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
    GridConf('21', BERT_MULTILINGUAL, seed=2),
    GridConf('22', BERT_MULTILINGUAL, seed=3),

    GridConf('23', RUGPT3_MEDIUM, seed=2),
    GridConf('24', RUGPT3_MEDIUM, seed=3),

    GridConf('25', RUGPT3_SMALL, seed=2),
    GridConf('26', RUGPT3_SMALL, seed=3),

    GridConf('27', RUGPT3_LARGE, seed=1),
    GridConf('28', RUGPT3_MEDIUM, seed=4),
    GridConf('29', RUGPT3_SMALL, seed=4),
    GridConf('30', RUGPT3_LARGE, seed=2),
]


#######
#   EXP
########


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


#######
#   PREDS
######


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


#####
#  SHOW
#######


def show_grid_scores(leaderboard, conf_task_scores, task_train_sizes,
                     tasks=TASKS, models=MODELS, confs=GRID_CONFS,
                     cols=3, width=2.5, height=2.5):
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
            for id, grid_task, score in conf_task_scores:
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


def show_seed_scores(leaderboard, conf_task_scores,
                     tasks=TASKS, models=MODELS, confs=GRID_CONFS):
    id_confs = {_.id: _ for _ in confs}
    model_task_scores = defaultdict(list)
    for id, task, score in conf_task_scores:
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


def show_seed_scores2(conf_task_scores,
                      tasks=TASKS, models=MODELS, confs=GRID_CONFS):
    id_confs = {_.id: _ for _ in confs}
    model_task_scores = defaultdict(list)
    for id, task, score in conf_task_scores:
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



####
#  TOP
#######


def select_top_conf(task, model, conf_task_scores, confs=GRID_CONFS):
    id, max_score = None, None
    id_confs = {_.id: _ for _ in confs}
    for conf_id, conf_task, score in conf_task_scores:
        score = score_value(score)
        conf = id_confs[conf_id]
        if conf.model == model and conf_task == task and (max_score is None or max_score < score):
            id = conf_id
            max_score = score
    return id


def find_grid_score(id, task, conf_task_scores):
    for conf_id, conf_task, score in conf_task_scores:
        if conf_id == id and conf_task == task:
            return score


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


######
#
#  DOCKER
#
######


DOCKERFILE = '''
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04

RUN apt-get update \\
  && apt-get install -y wget git

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \\
  && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \\
  && rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:$PATH

RUN git clone https://github.com/kuk/jiant-v1-legacy.git
WORKDIR jiant-v1-legacy
RUN git checkout russian-superglue

RUN conda env create -f environment.yml

# emulate conda activate
# conda shell.posix activate jiant
ENV PATH /opt/conda/envs/jiant/bin:$PATH
ENV CONDA_PREFIX "/opt/conda/envs/jiant"
ENV CONDA_DEFAULT_ENV "jiant"

# UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2
ENV LANG C.UTF-8
RUN pip install --no-dependencies -e .

WORKDIR ~
COPY transformers_cache exp/transformers_cache
COPY {task} exp/{task}
COPY main.py .
ENTRYPOINT ["python", "main.py", "infer", "exp", "{args_task}"]
'''

DOCKERIGNORE = '''
*
!transformers_cache
!{task}
!main.py
'''


def docker_build(exp_dir, task, name):
    args_task = task
    if task == LIDIRUS:
        task = TERRA

    path = join(exp_dir, 'Dockerfile')
    text = DOCKERFILE.format(
        task=task,
        args_task=args_task
    )
    dump_text(text, path)

    path = join(exp_dir, '.dockerignore')
    text = DOCKERIGNORE.format(task=task)
    dump_text(text, path)

    copy(__file__, exp_dir)

    try:
        command = ['docker', 'build', '-t', name, '.']
        log(f'Call docker: {command!r} in {exp_dir!r}')
        subprocess.run(command, cwd=exp_dir)
    finally:
        for filename in ['Dockerfile', '.dockerignore', 'main.py']:
            remove(join(exp_dir, filename))


def docker_login():
    subprocess.run([
        'docker', 'login',
        '--username', 'json_key',
        '--password', DOCKER_KEY,
        'cr.yandex'
    ])


def docker_push(image):
    remote = f'{DOCKER_REGISTRY}/{image}'
    subprocess.run(['docker', 'tag', image, remote])
    subprocess.run(['docker', 'push', remote])


def docker_pull(image):
    remote = f'{DOCKER_REGISTRY}/{image}'
    subprocess.run(['docker', 'pull', remote])
    subprocess.run(['docker', 'tag', remote, image])


def docker_find_pid(name):
    command = [
        'docker', 'inspect',
        '--format', '{{.State.Pid}}',
        name
    ]
    output = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding='utf8'
    ).stdout
    pid = output.strip()
    if pid:
        return pid


def retriable(function, *args, timeout=0.2, retries=10):
    for _ in range(retries):
        value = function(*args)
        if value is not None:
            return value
        sleep(timeout)


def show_docker_leaderboard(leaderboard, docker_scores, models=MODELS, tasks=TASKS):
    leaderboard_index = {
        (model, task): score
        for model, task, score in leaderboard
    }
    docker_scores_index = {
        (model, task): score
        for model, task, score in docker_scores
    }

    leaderboard_model_scores = Counter()
    docker_model_scores = Counter()
    count = len(tasks)
    for model in MODELS:
        for task in tasks:
            value = score_value(leaderboard_index[model, task])
            leaderboard_model_scores[model] += value / count
            value = score_value(docker_scores_index[model, task])
            docker_model_scores[model] += value / count

    SCORE = 'score'
    data = []
    for model in models:
        for task in [SCORE] + tasks:
            if task == SCORE:
                leaderboard_score = Score(leaderboard_model_scores[model])
                docker_score = Score(docker_model_scores[model])
            else:
                leaderboard_score = leaderboard_index[model, task]
                docker_score = docker_scores_index[model, task]
            better = score_value(docker_score) >= score_value(leaderboard_score)
            parts = [
                '✅' if better else '❌',
                format_score(leaderboard_score, sep='/'),
                format_score(docker_score, sep='/')
            ]
            value = ' '.join(parts)
            data.append([model, task, value])

    table = pd.DataFrame(data, columns=['model', 'task', 'value'])
    table = table.pivot(index='model', columns='task', values='value')
    table = table.reindex(index=models, columns=[SCORE] + tasks)
    table.columns.name = None
    table.index.name = None

    return table


######
#
#   PS
#
####


class PsStatsRecord(Record):
    __attributes__ = ['pid', 'cpu_usage', 'ram']


def ps_stats(pid):
    command = [
        'ps', '--no-headers', '-q', str(pid),
        '-o', '%cpu,rss'
    ]
    output = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        encoding='utf8'
    ).stdout

    # 0.2  18988
    parts = output.split()
    if not parts:
        return

    cpu_usage, ram = parts
    # 90.1 / 100 -> 0.900999999
    cpu_usage = round(float(cpu_usage) / 100, 4)
    ram = int(ram) * KB
    return PsStatsRecord(pid, cpu_usage, ram)


#######
#
#   NVIDIA
#
######


class NvidiaGPUStatsRecord(Record):
    __attributes__ = ['guid', 'total_gpu_ram', 'gpu_usage', 'gpu_ram_usage']


class NvidiaProcessStatsRecord(Record):
    __attributes__ = ['pid', 'guid', 'gpu_ram']


MIBS = {
    'KiB': KB,
    'MiB': MB,
    'GiB': GB
}


def parse_nvidia_gpu_ram(value):
    # 4443 MiB
    value, mib = value[:-3], value[-3:]
    value = value.strip()
    return int(float(value) * MIBS[mib])


def parse_nvidia_usage(value):
    # 22 %
    return float(value[:-2]) / 100


def parse_nvidia_gpu_stats(record):
    # memory.total [MiB], utilization.gpu [%], utilization.memory [%]
    # GPU-777aa4a9-8dac-a61b-5b5a-118d3e947546, 32510 MiB, 43 %, 22 %
    guid, total_gpu_ram, gpu_usage, gpu_ram_usage = record
    total_gpu_ram = parse_nvidia_gpu_ram(total_gpu_ram)
    gpu_usage = parse_nvidia_usage(gpu_usage)
    gpu_ram_usage = parse_nvidia_usage(gpu_ram_usage)
    return NvidiaGPUStatsRecord(guid, total_gpu_ram, gpu_usage, gpu_ram_usage)
    

def parse_nvidia_process_stats(record):
    # pid, gpu_uuid, used_gpu_memory [MiB]
    # 10042, GPU-777aa4a9-8dac-a61b-5b5a-118d3e947546, 4435 MiB
    pid, guid, gpu_ram = record
    gpu_ram = parse_nvidia_gpu_ram(gpu_ram)
    return NvidiaProcessStatsRecord(pid, guid, gpu_ram)


def parse_nvidia_output(output):
    lines = output.splitlines()
    records = parse_tsv(lines, sep=', ')
    next(records)
    return records


def nvidia_gpu_stats(guid):
    command = [
        'nvidia-smi', '--format=csv',
        '--query-gpu=gpu_uuid,memory.total,utilization.gpu,utilization.memory'
    ]
    output = subprocess.check_output(command, encoding='utf8')
    records = parse_nvidia_output(output)
    for record in records:
        record = parse_nvidia_gpu_stats(record)
        if record.guid == guid:
            return record


def nvidia_process_stats(pid):
    command = [
        'nvidia-smi', '--format=csv',
        '--query-compute-apps=pid,gpu_uuid,used_memory'
    ]
    output = subprocess.check_output(command, encoding='utf8')
    records = parse_nvidia_output(output)
    for record in records:
        record = parse_nvidia_process_stats(record)
        if record.pid == pid:
            return record


#####
#
#   BENCH
#
#####


class BenchRecord(Record):
    __attributes__ = [
        'timestamp',
        'cpu_usage', 'ram',
        'gpu_usage', 'gpu_ram',
    ]


def short_uid(cap=5):
    return str(uuid1())[:cap]


def bench(image, data_dir, task, input_size=10000, batch_size=128, delay=0.3):
    path = task_path(task, PUBLIC, VAL, dir=data_dir)
    lines = load_lines(path)
    lines = islice(cycle(lines), input_size)

    name = f'{image}_{short_uid()}'
    command = [
        'docker', 'run',
        '--gpus', 'all',
        '--interactive', '--rm',
        '--name', name,
        image,
        '--batch-size', str(batch_size)
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        universal_newlines=True
    )
    write_lines(lines, process.stdin)
    process.stdin.close()

    pid = retriable(docker_find_pid, name)
    if not pid:
        raise RuntimeError(f'pid not found, container {name!r}')

    record = BenchRecord()
    while process.poll() is None:
        stats = nvidia_process_stats(pid)
        if stats:
            _, guid, record.gpu_ram = nvidia_process_stats(pid)
            _, total_gpu_ram, record.gpu_usage, gpu_ram_usage = nvidia_gpu_stats(guid)

        stats = ps_stats(pid)
        if stats:  # process finished
            _, record.cpu_usage, record.ram = stats

        record.timestamp = time()
        yield record
        sleep(delay)


#######
#  REGISTRY
#########


class BenchRegistryRecord(Record):
    __attributes__ = ['model', 'task', 'input_size', 'batch_size', 'index']


def list_bench_registry(dir=BENCHES_DIR):
    for model in listdir(dir):
        for task in listdir(join(dir, model)):
            for filename in listdir(join(dir, model, task)):
                match = re.match(r'(\d+)_(\d+)_(\d+)\.jl', filename)
                if match:
                    input_size, batch_size, index = map(int, match.groups())
                    yield BenchRegistryRecord(
                        model, task,
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


def bench_path(model, task, input_size, batch_size, index=1, dir=BENCHES_DIR):
    return join(dir, model, task, f'{input_size}_{batch_size}_{index:02d}.jl')


def load_bench(path):
    lines = load_lines(path)
    items = parse_jl(lines)
    for item in items:
        yield BenchRecord.from_json(item)


def load_bench_registry(record):
    path = bench_path(*record)
    return list(load_bench(path))


########
#  SHOW
######


def show_benches(benches, width=8, height=6):
    tables = []
    for bench in benches:
        table = pd.DataFrame.from_records(bench)
        table.columns = BenchRecord.__attributes__
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


#######
#   STATS
#####


class BenchStats(Record):
    __attributes__ = ['total_time', 'gpu_time', 'max_gpu_ram']


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
        if record.gpu_usage and record.gpu_usage >= gpu_usage_treshold and previous:
            gpu_time += (record.timestamp - previous)
        previous = record.timestamp

    return BenchStats(total_time, gpu_time, max_gpu_ram)


#######
#   GROUP
#######


class BenchGroup(Record):
    __attributes__ = [
        'model', 'task', 'input_size',
        'gpu_rams', 'init_times', 'total_times', 'gpu_times'
    ]


def load_group_benches(registry, models=MODELS, tasks=TASKS,
                       input_size=2000, batch_size=32):
    for model in models:
        for task in tasks:
            records = query_bench_registry(
                registry,
                input_size=1,
                batch_size=1,
                model=model,
                task=task
            )
            benches = (load_bench_registry(_) for _ in records)
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
            benches = (load_bench_registry(_) for _ in records)
            stats = [bench_stats(_) for _ in benches]

            total_times = [_.total_time for _  in stats]
            gpu_times = [_.gpu_time for _ in stats]

            yield BenchGroup(
                model, task, input_size,
                gpu_rams, init_times, total_times, gpu_times
            )


#######
#   REPORT
#######


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


    palette = {
        RUBERT: 'blue',
        RUBERT_CONVERSATIONAL: 'blue',
        BERT_MULTILINGUAL: 'green',

        RUGPT3_SMALL: 'red',
        RUGPT3_MEDIUM: 'red',
        RUGPT3_LARGE: 'red',
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
    ax.scatter(xs, ys, c=colors, alpha=0.5)

    labels = [
        ','.join(_)
        for _ in samples
    ]
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.yaxis.set_ticks_position('right')

    ax.set_xticks([])
    ax.set_xlabel('(rps - min) / (max - min)')

    fig.set_size_inches(6, 10)


#####
#
#   SCORE PERF
#
#####


def show_score_perf(docker_scores, leaderboard, bench_groups,
                    models=MODELS, tasks=TASKS,
                    width=6, height=6):
    leaderboard_index = {
        (model, task): score
        for model, task, score in leaderboard
    }
    docker_scores_index = {
        (model, task): score
        for model, task, score in docker_scores
    }

    data = {}
    for model in models:
        for task in tasks:
            # fill in rucos, muserc
            score = docker_scores_index.get(
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



#######
#
#   CLI
#
######


def cli_infer(args):
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


def cli_docker_build(args):
    docker_build(args.exp_dir, args.task, args.name)


def cli_docker_login(args):
    docker_login()


def cli_docker_push(args):
    docker_push(args.image)


def cli_docker_pull(args):
    docker_pull(args.image)


def cli_eval(args):
    preds = list(load_jl(args.preds))
    targets = list(load_jl(args.targets))
    metrics = eval(args.task, preds, targets)
    print(json.dumps(metrics, indent=2))


def cli_bench(args):
    log(
        f'Bench {args.image!r}, input_size={args.input_size}, '
        f'batch_size={args.batch_size}'
    )
    records = bench(
        args.image, args.data_dir, args.task,
        input_size=args.input_size,
        batch_size=args.batch_size
    )
    items = (_.as_json for _ in records)
    lines = format_jl(items)
    for line in lines:
        print(line, flush=True)


def existing_path(path):
    if not exists(path):
        raise argparse.ArgumentTypeError(f'{path!r} does not exist')
    return path


def main(args):
    parser = argparse.ArgumentParser(prog='main.py')
    parser.set_defaults(function=None)
    subs = parser.add_subparsers()

    sub = subs.add_parser('infer')
    sub.set_defaults(function=cli_infer)
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

    docker = subs.add_parser('docker').add_subparsers()

    sub = docker.add_parser('build')
    sub.set_defaults(function=cli_docker_build)
    sub.add_argument('exp_dir', type=existing_path)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('name')

    sub = docker.add_parser('login')
    sub.set_defaults(function=cli_docker_login)

    sub = docker.add_parser('push')
    sub.set_defaults(function=cli_docker_push)
    sub.add_argument('image')

    sub = docker.add_parser('pull')
    sub.set_defaults(function=cli_docker_pull)
    sub.add_argument('image')

    sub = subs.add_parser('eval')
    sub.set_defaults(function=cli_eval)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('preds', type=existing_path)
    sub.add_argument('targets', type=existing_path)

    sub = subs.add_parser('bench')
    sub.set_defaults(function=cli_bench)
    sub.add_argument('image')
    sub.add_argument('data_dir', type=existing_path)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('--input-size', type=int, default=10000)
    sub.add_argument('--batch-size', type=int, default=128)

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
    
