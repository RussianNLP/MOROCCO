
import sys
import json
import argparse
import warnings
from datetime import datetime
from os.path import (
    dirname,
    exists,
)

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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


#####
#
#  IO
#
#####


def load_lines(path):
    with open(path) as file:
        for line in file:
            yield line.rstrip('\n')


def parse_jsonl(lines):
    for line in lines:
        yield json.loads(line)


def format_json(item):
    return json.dumps(item, ensure_ascii=False)


def format_jsonl(items):
    for item in items:
        yield format_json(item)


def load_jsonl(path):
    lines = load_lines(path)
    return parse_jsonl(lines)


def load_pickle(path):
    with warnings.catch_warnings():
        # Trying to unpickle estimator TfidfTransformer from version
        # 0.21.3 when using version 1.1.1
        warnings.simplefilter('ignore')

        return joblib.load(path)


def dump_pickle(object, path):
    joblib.dump(object, path)


#######
#
#   TRAIN
#
######


def terra_encode(item):
    premise = str(item['premise']).strip()
    hypothesis = item['hypothesis']
    label = item.get('label')
    text = f'{premise} {hypothesis}'
    return text, label


def danetqa_encode(item):
    text = str(item['question']).strip()
    label = item.get('label')
    return text, label


def lidirus_encode(item):
    premise = str(item['sentence1']).strip()
    hypothesis = item['sentence2']
    label = item.get('label')
    text = f'{premise} {hypothesis}'
    return text, label


def parus_encode(item):
    premise = str(item['premise']).strip()
    choice1 = item['choice1']
    choice2 = item['choice2']
    label = item.get('label')
    question = (
        'Что было ПРИЧИНОЙ этого?'
        if item['question'] == 'cause'
        else 'Что случилось в РЕЗУЛЬТАТЕ?'
    )
    text = f'{premise} {question} {choice1} {choice2}'
    return text, label


def rcb_encode(item):
    premise = str(item['premise']).strip()
    hypothesis = item['hypothesis']
    label = item.get('label')
    text = f'{premise} {hypothesis}'
    return text, label


def russe_encode(item):
    sentence1 = item['sentence1'].strip()
    sentence2 = item['sentence2'].strip()
    word = item['word'].strip()
    label = item.get('label')
    text = f'{sentence1} {sentence2} {word}'
    return text, label


def rwsd_encode(item):
    premise = str(item['text']).strip()
    span1 = item['target']['span1_text']
    span2 = item['target']['span2_text']
    label = item.get('label')
    text = f'{premise} {span1} {span2}'
    return text, label


TASK_ENCODERS = {
    DANETQA: danetqa_encode,
    LIDIRUS: lidirus_encode,
    PARUS: parus_encode,
    RCB: rcb_encode,
    RUSSE: russe_encode,
    RWSD: rwsd_encode,
    TERRA: terra_encode,
}


def encode(items, encode_item, tfidf_vectorizer):
    ids, texts, labels = [], [], []
    for item in items:
        id = item['idx']
        text, label = encode_item(item)

        ids.append(id)
        texts.append(text)
        labels.append(label)

    X = tfidf_vectorizer.transform(texts)
    return ids, X, labels


def fit_logreg(X, labels):
    classifier = LogisticRegression()
    return classifier.fit(X, labels)


######
#
#   INFER
#
#######


def infer(ids, X, classifier):
    preds = classifier.predict(X)
    for id, label in zip(ids, preds):
        yield {
            'idx': id,
            'label': label.item()
        }


######
#
#  SCORE
#
#######


def id_labels(items):
    return {
        _['idx']: _['label']
        for _ in items
    }


def score(task, preds, targets):
    pred_id_labels = id_labels(preds)
    target_id_labels = id_labels(targets)

    pred_labels = []
    target_labels = []
    for id in pred_id_labels.keys() & target_id_labels.keys():
        pred_labels.append(pred_id_labels[id])
        target_labels.append(target_id_labels[id])

    accuracy = accuracy_score(target_labels, pred_labels)
    return {
        'accuracy': accuracy
    }


#######
#
#   CLI
#
######


def log(message):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(
        '[%s] %s' % (now, message),
        file=sys.stderr,
        flush=True
    )


def cli_train(args):
    log(f'Load TF-IDF vectorizer {args.tfidf_vectorizer_path!r}')
    tfidf_vectorizer = load_pickle(args.tfidf_vectorizer_path)

    log('Reading train from stdin')
    items = parse_jsonl(sys.stdin)
    encode_item = TASK_ENCODERS[args.task]
    _, X, labels = encode(items, encode_item, tfidf_vectorizer)

    log('Train classifier')
    classifier = fit_logreg(X, labels)

    log(f'Save classifier {args.classifier_path!r}')
    dump_pickle(classifier, args.classifier_path)


def cli_infer(args):
    log(f'Load TF-IDF vectorizer {args.tfidf_vectorizer_path!r}')
    tfidf_vectorizer = load_pickle(args.tfidf_vectorizer_path)

    log(f'Load classifier {args.classifier_path}')
    classifier = load_pickle(args.classifier_path)

    log('Reading test from stdin')
    items = parse_jsonl(sys.stdin)
    encode_item = TASK_ENCODERS[args.task]
    ids, X, _ = encode(items, encode_item, tfidf_vectorizer)

    log('Run classifier')
    preds = infer(ids, X, classifier)
    for line in format_jsonl(preds):
        print(line)


def cli_score(args):
    log(f'Load preds {args.preds_path!r}')
    preds = load_jsonl(args.preds_path)

    log(f'Load targets {args.targets_path!r}')
    targets = load_jsonl(args.targets_path)

    metrics = score(args.task, preds, targets)
    print(format_json(metrics))


def existing_path(path):
    if not exists(path):
        raise argparse.ArgumentTypeError(f'{path!r} does not exist')
    return path


def existing_parent(path):
    existing_path(dirname(path))
    return path


def main(args):
    parser = argparse.ArgumentParser(prog='main.py')
    parser.set_defaults(function=None)
    subs = parser.add_subparsers()

    sub = subs.add_parser('train')
    sub.set_defaults(function=cli_train)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('tfidf_vectorizer_path', type=existing_path)
    sub.add_argument('classifier_path', type=existing_parent)

    sub = subs.add_parser('infer')
    sub.set_defaults(function=cli_infer)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('tfidf_vectorizer_path', type=existing_path)
    sub.add_argument('classifier_path', type=existing_parent)

    sub = subs.add_parser('score')
    sub.set_defaults(function=cli_score)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('preds_path', type=existing_path)
    sub.add_argument('targets_path', type=existing_path)

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
