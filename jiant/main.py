
import re
import sys
import json
import logging
import argparse
import subprocess
from contextlib import contextmanager
from datetime import datetime
from fnmatch import fnmatch
from os.path import (
    join,
    exists,
    isdir
)
from os import (
    environ,
    remove,
    rename,
    makedirs,
    listdir,
    rmdir,
)
from shutil import (
    copy,
    copytree,
    rmtree
)
from importlib import reload
from tempfile import TemporaryDirectory

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
MODEL_HUB_NAMES = {
    RUBERT_CONVERSATIONAL: 'DeepPavlov/rubert-base-cased-conversational',
    RUBERT: 'DeepPavlov/rubert-base-cased',
    BERT_MULTILINGUAL: 'bert-base-multilingual-cased',

    RUGPT3_LARGE: 'sberbank-ai/rugpt3large_based_on_gpt2',
    RUGPT3_MEDIUM: 'sberbank-ai/rugpt3medium_based_on_gpt2',
    RUGPT3_SMALL: 'sberbank-ai/rugpt3small_based_on_gpt2',
}

JIANT_DIR = 'jiant-v1-legacy'
JIANT_CONF = join(JIANT_DIR, 'jiant/config/superglue_bert.conf')


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


#######
#
#   PATH
#
####


def maybe_mkdir(dir):
    makedirs(dir, exist_ok=True)


def maybe_copytree(source, target):
    if not exists(target):
        copytree(source, target)


def rm_any(path):
    if isdir(path):
        rmtree(path)
    else:
        remove(path)


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


def parse_jsonl(lines):
    for line in lines:
        yield json.loads(line)


def format_jsonl(items):
    for item in items:
        yield json.dumps(item, ensure_ascii=False)


def load_jsonl(path):
    lines = load_lines(path)
    return parse_jsonl(lines)


def dump_jsonl(items, path):
    lines = format_jsonl(items)
    dump_lines(lines, path)


def dump_text(text, path):
    with open(path, 'w') as file:
        file.write(text)


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
            f'pretrain_tasks = {task}, target_tasks = {target_tasks}, '
            'do_pretrain = 1, do_target_task_training = 0, do_full_eval = 1, '
            f'batch_size = 4, val_interval = 100, {spec}'
        ])


def patch_exp_params(params, model):
    keys = ['data_dir', 'exp_dir', 'local_log_path', 'project_dir', 'run_dir']
    for key in keys:
        params[key] = None

    # by default auto, whyyy?
    params.pool_type = 'max'

    # by default tokenizer=auto, why?
    params.tokenizer = MODEL_HUB_NAMES[model]


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
            name = re.match(r'([^_\.]+)', item).group(1)
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


RWSD_ITEM = {
    'idx': 0,
    'target': {
        'span1_text': 'Члены городского совета',
        'span2_text': 'они опасались',
        'span1_index': 0,
        'span2_index': 10
    },
    'label': True,
    'text': (
        'Члены городского совета отказали организаторам митинга '
        'в разрешении, потому что они опасались насилия.'
    )
}


def dump_task(data_dir, task, items):
    title = TASK_TITLES[task]
    dir = join(data_dir, title)
    maybe_mkdir(dir)

    if task == LIDIRUS:
        path = join(dir, title + '.jsonl')
        dump_jsonl(items, path)

    else:
        path = join(dir, 'test.jsonl')
        dump_jsonl(items, path)

        for filename in ['train.jsonl', 'val.jsonl']:
            data = []
            if task == RWSD:
                # rwsd load_data breaks on empty train
                data = [RWSD_ITEM]

            path = join(dir, filename)
            dump_jsonl(data, path)

    return dir


def load_preds(dir, task):
    path = join(dir, TASK_TITLES[task] + '.jsonl')
    return load_jsonl(path)


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


######
#
#  DOCKER BUILD
#
######


DOCKERFILE = '''
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
# https://github.com/NVIDIA/nvidia-docker/issues/1631#issuecomment-1124509101
RUN apt-key adv --fetch-keys \\
  https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \\
  && apt-get update \\
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
# https://github.com/allenai/specter/issues/27
RUN pip install overrides==3.1.0
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


#######
#
#   CLI
#
######


def cli_train(args):
    train_jiant(
        args.model, args.task,
        args.exps_dir, args.data_dir,
        seed=args.seed
    )
    strip_exp(args.exps_dir, args.model, args.task)


def cli_infer(args):
    log('Reading items from stdin')
    items = list(parse_jsonl(sys.stdin))
    log(f'Read {len(items)} items')

    preds = infer_jiant(
        args.exp_dir, args.task, items,
        batch_size=args.batch_size
    )

    log('Writing preds to stdout')
    lines = format_jsonl(preds)
    for line in lines:
        print(line)


def cli_eval(args):
    preds = list(load_jsonl(args.preds))
    targets = list(load_jsonl(args.targets))
    metrics = eval(args.task, preds, targets)
    print(json.dumps(metrics, indent=2))


def cli_docker_build(args):
    docker_build(args.exp_dir, args.task, args.name)


def print_jsonl(items):
    lines = format_jsonl(items)
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

    sub = subs.add_parser('train')
    sub.set_defaults(function=cli_train)
    sub.add_argument('model', choices=MODELS)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('exps_dir')
    sub.add_argument('data_dir', type=existing_path)
    sub.add_argument('--seed', type=int, default=1)

    sub = subs.add_parser('infer')
    sub.set_defaults(function=cli_infer)
    sub.add_argument('exp_dir', type=existing_path)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('--batch-size', type=int, default=128)

    sub = subs.add_parser('eval')
    sub.set_defaults(function=cli_eval)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('preds', type=existing_path)
    sub.add_argument('targets', type=existing_path)

    sub = subs.add_parser('docker-build')
    sub.set_defaults(function=cli_docker_build)
    sub.add_argument('exp_dir', type=existing_path)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('name')

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
