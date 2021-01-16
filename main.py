
import re
import sys
from datetime import datetime
from os import (
    environ,
    mkdir,
    makedirs,
    listdir
)
from os.path import (
    join,
    exists,
    dirname,
    expanduser
)
from shutil import (
    copy,
    copytree
)
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
    defaultdict
)

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
EXPS = [
    RUBERT,
    RUBERT_CONVERSATIONAL,
    BERT_MULTILINGUAL
]
EXP_HUB_NAMES = {
    RUBERT_CONVERSATIONAL: 'DeepPavlov/rubert-base-cased-conversational',
    RUBERT: 'DeepPavlov/rubert-base-cased',
    BERT_MULTILINGUAL: 'bert-base-multilingual-cased',
}

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


#####
#
#  S3
#
#####


def s3_call(args, key_id=S3_KEY_ID, key=S3_KEY,
            region=S3_REGION, endpoint=S3_ENDPOINT):
    with env(
        AWS_ACCESS_KEY_ID=key_id,
        AWS_SECRET_ACCESS_KEY=key
    ):
        command = ['aws', '--region', region, '--endpoint-url', endpoint, 's3']
        return subprocess.run(
            command + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )


def s3_path(path, bucket=S3_BUCKET):
    return f's3://{bucket}/{path}'


def s3_sync(source, target):
    return s3_call(['sync', source, target])


#######
#
#   TRAIN
#
######


def train_jiant(exp, task, exps_dir, data_dir, config=JIANT_CONF, seed=1):
    target_tasks = task
    if task == TERRA:
        target_tasks = f'"{TERRA},{LIDIRUS}"'

    input_module = EXP_HUB_NAMES[exp]

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
            f'input_module = {input_module}, exp_name = {exp}, '
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


def patch_exp_params(params, exp):
    for key in ['data_dir', 'exp_dir', 'local_log_path', 'project_dir', 'run_dir']:
        params[key] = None

    params.pool_type = 'max'  # by default auto, whyyy?
    params.tokenizer = EXP_HUB_NAMES[exp]  # by default tokenizer=auto, why?


def best_model_path(dir):
    for filename in listdir(dir):
        if fnmatch(filename, 'model_*.best.th'):
            return join(dir, filename)


def copy_exp(exps_dir, target_dir, exp, tasks):
    source_dir = join(exps_dir, exp)
    makedirs(target_dir)        

    # assert no extra data in cache
    log(f'Copy transformers cache')
    copytree(
        join(source_dir, 'transformers_cache'),
        join(target_dir, 'transformers_cache')
    )

    for task in tasks:
        if task == LIDIRUS:
            # use terra model
            continue

        log(f'Copy {task!r} tasks')
        mkdir(join(target_dir, task))

        path = join(source_dir, task, 'params.conf')
        params = params_from_file(path)
        patch_exp_params(params, exp)
        write_params(
            params,
            join(target_dir, task, 'params.conf')
        )

        copy(
            best_model_path(join(source_dir, task)),
            join(target_dir, task, 'model.th')
        )
        copy(
            join(source_dir, task, 'log.log'),
            join(target_dir, task, 'log.log')
        )


#######
#
#   INFER
#
#######


RWSD_ITEM = {'idx': 0, 'target': {'span1_text': 'Члены городского совета', 'span2_text': 'они опасались', 'span1_index': 0, 'span2_index': 10}, 'label': True, 'text': 'Члены городского совета отказали организаторам митинга в разрешении, потому что они опасались насилия.'}


def dump_task(data_dir, task, items):
    title = TASK_TITLES[task]
    dir = join(data_dir, title)
    makedirs(dir, exist_ok=True)

    if task == LIDIRUS:
        path = join(dir, title + '.jsonl')
        lines = format_jl(items)
        dump_lines(lines, path)
        
    else:
        path = join(dir, 'test.jsonl')
        lines = format_jl(items)
        dump_lines(lines, path)

        for filename in ['train.jsonl', 'val.jsonl']:
            data = []
            if task == RWSD:
                # rwsd load_data breaks on empty train
                data = [RWSD_ITEM]

            path = join(dir, filename)
            lines = format_jl(data)
            dump_lines(lines, path)

    return dir


def load_preds(dir, task):
    path = join(dir, TASK_TITLES[task] + '.jsonl')
    lines = load_lines(path)
    return parse_jl(lines)


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
#   CLI
#
######


def main(args):
    if len(args) != 1:
        log('Set task name')
        return

    task, = args
    if task not in TASKS:
        log(f'Unkonwn task {task!r}')
        return

    log('Reading items from stdin')
    items = list(parse_jl(sys.stdin))
    log(f'Read {len(items)} items')

    preds = infer_jiant('exp', task, items)
    
    log('Writing preds to stdout')
    lines = format_jl(preds)
    for line in lines:
        print(line)


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
        exp = LEADERBOARD_RENAMES[record[name_offset]]
        scores = [parse_leaderboard_score(_) for _ in record[scores_offset:]]
        yield exp, dict(zip(tasks, scores))


if __name__ == '__main__':
    main(sys.argv[1:])
