
import sys
from dataclasses import (
    dataclass,
    asdict
)
from datetime import datetime
from time import (
    time,
    sleep
)
from os import (
    stat,
    makedirs,
    remove,
)
from os.path import (
    join,
    exists,
    expanduser,
    isdir
)
from shutil import (
    copytree,
    rmtree
)
from uuid import uuid1
import json
import subprocess

from itertools import (
    cycle,
    islice
)

import argparse


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

PROJ_DIR = expanduser('~')
PROJ_DIR = '..'

TRAIN = 'train'
TEST = 'test'
VAL = 'val'

EXPS_DIR = join(PROJ_DIR, 'exps')
GRID_PREDS_DIR = join(PROJ_DIR, 'preds/grid')
CONF_TASK_METRICS = join(GRID_PREDS_DIR, 'conf_task_metrics.jl')
BENCHES_DIR = join(PROJ_DIR, 'benches')


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


def rm_any(path):
    if isdir(path):
        rmtree(path)
    else:
        remove(path)


def path_modified(path):
    record = stat(path)
    return datetime.fromtimestamp(record.st_mtime)


#######
#
#  DOCKER
#
######


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
        return int(pid)


def retriable(function, *args, timeout=0.2, retries=10):
    for _ in range(retries):
        value = function(*args)
        if value is not None:
            return value
        sleep(timeout)


######
#
#   PS
#
####


@dataclass
class PsStatsRecord:
    pid: str
    cpu_usage: float
    ram: int


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


@dataclass
class NvidiaGPUStatsRecord:
    guid: str
    total_gpu_ram: int
    gpu_usage: float
    gpu_ram_usage: int


@dataclass
class NvidiaProcessStatsRecord:
    pid: str
    guid: str
    gpu_ram: int


KB = 1024
MB = 1024 * KB
GB = 1024 * MB

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
    pid = int(pid)
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


@dataclass
class BenchRecord:
    timestamp: datetime
    cpu_usage: float = None
    ram: int = None
    gpu_usage: float = None
    gpu_ram: int = None


def task_path(dir, task, split):
    title = TASK_TITLES[task]

    if task == LIDIRUS:
        name = title
    else:
        name = split

    return join(dir, title, f'{name}.jsonl')


def bench_input(dir, task, size):
    path = task_path(dir, task, VAL)
    lines = load_lines(path)
    return islice(cycle(lines), size)


def probe_pid(pid):
    record = BenchRecord(
        timestamp=time()
    )

    stats = nvidia_process_stats(pid)
    if stats:
        record.gpu_ram = stats.gpu_ram

        # via nvidia-smi can not get both gpu ram and usage in one
        # call
        stats = nvidia_gpu_stats(stats.guid)
        record.gpu_usage = stats.gpu_usage

    stats = ps_stats(pid)
    if stats:
        record.cpu_usage = stats.cpu_usage
        record.ram = stats.ram

    return record


def short_uid(cap=5):
    return str(uuid1())[:cap]


def gen_name(image):
    # slash not allowed
    # russiannlp/rubert-parus -> russiannlp_rubert-parus
    name = image.replace('/', '_')

    # run in parallel
    uid = short_uid()
    return f'{name}_{uid}'


def bench_docker(
        image, data_dir, task,
        input_size=10000,
        batch_size=128,
        delay=0.3
):
    lines = bench_input(data_dir, task, input_size)
    name = gen_name(image)
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

    while process.poll() is None:
        yield probe_pid(pid)
        sleep(delay)


#######
#
#   CLI
#
######


def cli_bench(args):
    log(
        f'Bench {args.image!r}, input_size={args.input_size}, '
        f'batch_size={args.batch_size}'
    )
    records = bench_docker(
        args.image, args.data_dir, args.task,
        input_size=args.input_size,
        batch_size=args.batch_size
    )
    items = (asdict(_) for _ in records)
    print_jl(items)


def print_jl(items):
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