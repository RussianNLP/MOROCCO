from rsg_funcs import *

import json
import os, sys
import torch
import argparse

from transformers import AutoTokenizer, GPT2LMHeadModel

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


def format_json(item):
    return json.dumps(item, ensure_ascii=False)


def format_jsonl(items):
    for item in items:
        yield format_json(item)


def infer(args):
    task_dict = {DANETQA: danetqa_get_answer,
                 LIDIRUS: lidirus_get_answer,
                 MUSERC: muserc_get_answer,
                 PARUS: parus_get_answer,
                 RCB: rcb_get_answer,
                 RUCOS: rucos_get_answer,
                 RUSSE: russe_get_answer,
                 RWSD: rwsd_get_answer,
                 TERRA: terra_get_answer}
                 
    # device = 'cuda' if torch.cuda.is_available else 'cpu'
    device = torch.device(args.device)
    
    bench_path =  os.path.join("/workspace", args.model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(bench_path, use_fast=True)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})

    model = GPT2LMHeadModel.from_pretrained(bench_path)
    model.config.use_cache = True
    model.to(device)

    data = [json.loads(line) for line in sys.stdin.readlines()]

    # with open(os.path.join(bench_path,'bench_params.json'), 'r') as json_file:
    #     bench_params = json.load(json_file)
    # task = bench_params["task"]

    preds = task_dict[args.task](data, args.batch_size, tokenizer, model, device)

    for line in format_jsonl(preds):
        print(line) 


def main(args):
    parser = argparse.ArgumentParser(prog='main.py')
    parser.set_defaults(function=None)
    subs = parser.add_subparsers()

    sub = subs.add_parser('infer')
    sub.set_defaults(function=infer)
    sub.add_argument('--task', choices=TASKS)
    sub.add_argument('--model-path', type=str)
    sub.add_argument('--batch-size', type=int)
    sub.add_argument('--device', type=str, default='cpu')

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
    
