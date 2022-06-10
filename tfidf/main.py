
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


#########
#
#   RUCOS
#
#######


def infer_rucos(items, tfidf_vectorizer):
    # {
    #   "idx": 1,
    #   "passage": {
    #     "text": "Израильская авиация реагирует на ракетный и минометный обстрел, устроенный боевиками ХАМАС. В ночь на среду воздушные удары нанесены по 25 военным объектам палестинского радикального движения. Израильская авиация в ответ на ракетный и минометный обстрел с территории сектора Газа, в ночь на среду, Мир в дыму и огне. Это не просто кадры новостной хроники о ситуации в секторе Газа.\n@highlight\nВ Швеции задержаны двое граждан РФ в связи с нападением на чеченского блогера\n@highlight\nТуризм в эпоху коронавируса: куда поехать? И ехать ли вообще?\n@highlight\nКомментарий: Россия накануне эпидемии - виноватые назначены заранее",
    #     "entities": [
    #       {
    #         "start": 85,
    #         "end": 90
    #       },
    #       {
    #         "start": 275,
    #         "end": 279
    #       },
    #       {
    #         "start": 397,
    #         "end": 404
    #       },
    #     ]
    #   },
    #   "qas": [
    #     {
    #       "query": "Однако @placeholder и Израиль никак не прокомментировали это сообщение.",
    #       "answers": [
    #         {
    #           "start": 85,
    #           "end": 90,
    #           "text": "ХАМАС"
    #         },
    #         {
    #           "start": 488,
    #           "end": 493,
    #           "text": "ХАМАС"
    #         }
    #       ],
    #       "idx": 1
    #     }
    #   ]
    # }

    for item in items:
        passage = item['passage']
        text = passage['text']

        entities = [
            text[_['start']:_['end']]
            for _ in passage['entities']
        ]

        text = text.replace('\n@highlight\n', ' ')
        text_emb = tfidf_vectorizer.transform([text])

        parts = []
        for query in item['qas']:
            query_text = query['query']
            options = [
                query_text.replace('@placeholder', _)
                for _ in entities
            ]
            options_emb = tfidf_vectorizer.transform(options)
            similarities = cosine_similarity(text_emb, options_emb)

            index = similarities.argmax()
            part = entities[index]
            parts.append(part)

        label = ' '.join(parts)
        yield {
            'idx': item['idx'],
            'label': label
        }


######
#
#   MUSERC
#
####


def infer_muserc(items, tfidf_vectorizer):
    # {
    #   "idx": 0,
    #   "passage": {
    #     "text": "(1) Самый первый «остров» Архипелага возник в 1923 году на месте Соловецкого монастыря. (2) Затем появились ТОНы — тюрьмы особого назначения и этапы. (3) Люди попадали на Архипелаг разными сп особами: в вагон-заках, на баржах, пароходах и пешими этапами. (4) В тюрьмы арестованных доставляли в «воронках» — фургончиках чёрного цвета. (5) Роль портов Архипелага играли пересылки, временные лагеря, состоящие из палаток, землянок, бараков или участков земли под открытым небом. (6) На всех пересылках держать «политических» в узде помогали специально отобранные урки, или «социально близкие». (7) Солже ницын побывал на пересылке Красная Пресня в 1945 году. (8) Эмигранты, крестьяне и «малые народы» перевозили красными эшелонами. (9) Чаще всего такие эшелоны останав­ливались на пустом месте, посреди степи ли тайги, и осуждённые сами строили лагерь. (10) Особо важные заключённые, в основном учёные, перевозились спецконвоем. (11) Так перевозили и Солженицына. (12) Он назвался ядерным физиком, и после Красной Пресни его перевезли в Бутырки.",
    #     "questions": [
    #       {
    #         "question": "Почему Солженицына перевозили спецконвоем?",
    #         "answers": [
    #           {
    #             "idx": 0,
    #             "text": "Так перевозили особо важных заключенных.",
    #             "label": 1
    #           },
    #           {
    #             "idx": 1,
    #             "text": "Потому, что был эмигрантом.",
    #             "label": 0
    #           },
    #           ...
    #         ],
    #         "idx": 0
    #       },
    #       {
    #         "question": "Как люди попадали в тюрьмы особого типа на Соловках?",
    #         "answers": [
    #           {
    #             "idx": 5,
    #             "text": "Люди попадали на архипелаг с помощью дрезин и вертолётов.",
    #             "label": 0
    #           },
    #           {
    #             "idx": 6,
    #             "text": "Люди попадали на Архипелаг разными способами: в вагон-заках, на баржах, пароходах, пешими этапами, а также спецконвоем.",
    #             "label": 1
    #           },
    #           ...
    #         ],
    #         "idx": 1
    #       }
    #     ]
    #   }
    # }

    for item in items:
        passage = item['passage']
        text = passage['text']
        text_emb = tfidf_vectorizer.transform([text])

        question_preds = []
        for question in passage['questions']:
            question_text = question['question']
            answers = question['answers']

            options = []
            for answer in answers:
                answer_text = answer['text']
                option = f'{question_text} {answer_text}'
                options.append(option)

            options_emb = tfidf_vectorizer.transform(options)
            similarity = cosine_similarity(text_emb, options_emb)

            top = similarity.argsort().flatten()[-2:]
            answer_preds = []
            for index, answer in enumerate(answers):
                label = int(index in top)
                pred = {
                    'idx': answer['idx'],
                    'label': label
                }
                answer_preds.append(pred)

            pred = {
                'idx': question['idx'],
                'answers': answer_preds
            }
            question_preds.append(pred)

        # {
        #   "idx": 0,
        #   "passage": {
        #     "questions": [
        #       {
        #         "idx": 0,
        #         "answers": [
        #           {
        #             "idx": 0,
        #             "label": 0
        #           },
        #           {
        #             "idx": 1,
        #             "label": 0
        #           },
        #           ...
        #         ]
        #       },
        #       {
        #         "idx": 1,
        #         "answers": [
        #           {
        #             "idx": 4,
        #             "label": 0
        #           },
        #           {
        #             "idx": 5,
        #             "label": 1
        #           },
        #           ...
        #         ]
        #       },
        #       ...

        yield {
            'idx': item['idx'],
            'passage': {
                'questions': question_preds
            }
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

    log('Reading test from stdin')
    items = parse_jsonl(sys.stdin)

    if args.task == MUSERC:
        preds = infer_muserc(items, tfidf_vectorizer)

    elif args.task == RUCOS:
        preds = infer_rucos(items, tfidf_vectorizer)

    else:
        log(f'Load classifier {args.classifier_path}')
        classifier = load_pickle(args.classifier_path)

        encode_item = TASK_ENCODERS[args.task]
        ids, X, _ = encode(items, encode_item, tfidf_vectorizer)

        log('Run classifier')
        preds = infer(ids, X, classifier)

    for line in format_jsonl(preds):
        print(line)


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
    sub.add_argument('classifier_path', type=existing_parent, nargs='?')

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
