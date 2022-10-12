
import re
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
import statistics

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


def dump_jsonl(items, path):
    lines = format_jsonl(items)
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
    timestamp: float
    cpu_usage: float
    ram: int
    gpu_usage: float
    gpu_ram: int


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
    cpu_usage, ram = None, None
    gpu_usage, gpu_ram = None, None

    stats = ps_stats(pid)
    if stats:
        cpu_usage = stats.cpu_usage
        ram = stats.ram

    stats = nvidia_process_stats(pid)
    if stats:
        gpu_ram = stats.gpu_ram

        # via nvidia-smi can not get both gpu ram and usage in one
        # call
        stats = nvidia_gpu_stats(stats.guid)
        gpu_usage = stats.gpu_usage

    return cpu_usage, ram, gpu_usage, gpu_ram


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
        delay=0.3,
        device='cpu',
        model_path=None,
        volume_path=None
):
    lines = bench_input(data_dir, task, input_size)
    name = gen_name(image)
    command = [
        'docker', 'run',
        '--volume', volume_path+':/workspace',
        '--interactive', '--rm',
        '--runtime', 'nvidia', 
        '--name', name,
        image,
        '--batch-size', str(batch_size),
        '--task', task,
        '--model-path', model_path,
        '--device', device
    ] # '--gpus', 'all',
    log(' '.join(command))
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
        cpu_usage, ram, gpu_usage, gpu_ram = probe_pid(pid)
        timestamp = time()

        yield BenchRecord(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            ram=ram,
            gpu_usage=gpu_usage,
            gpu_ram=gpu_ram
        )
        sleep(delay)


########
#
#   PLOT
#
#####


@dataclass
class Bench:
    path: str
    task: str
    input_size: int
    batch_size: int
    records: [BenchRecord]


def parse_bench_path(path):
    match = re.search(r'([^/]+)/(\d+)_(\d+)_\d+\.jsonl$', path)
    if match:
        task, input_size, batch_size = match.groups()
        return task, int(input_size), int(batch_size)

    raise ValueError(f'bad path {path!r}')


def load_bench(path):
    task, input_size, batch_size = parse_bench_path(path)

    items = load_jsonl(path)
    records = [
        BenchRecord(**_)
        for _ in items
    ]

    return Bench(
        path,
        task, input_size, batch_size,
        records
    )


def plot_benches(benches, width=8, height=7):
    # speed up main.py launch
    import pandas as pd

    from matplotlib import patches
    from matplotlib import pyplot as plt

    group_tables = []
    for bench in benches:
        table = pd.DataFrame(bench.records)
        group = f'{bench.input_size}_{bench.batch_size}_*.jsonl'
        group_tables.append([group, table])

    keys = ['cpu_usage', 'ram', 'gpu_usage', 'gpu_ram']
    lims = {
        key: max(table[key].max() for _, table in group_tables)
        for key in keys
    }
    groups = sorted({group for group, _ in group_tables})

    fig, axes = plt.subplots(len(keys), 1, sharex=True)
    axes = axes.flatten()

    get_color = plt.cm.get_cmap('rainbow', len(groups))
    group_colors = {
        group: get_color(index)
        for index, group in enumerate(groups)
    }

    for key, ax in zip(keys, axes):
        for group, table in group_tables:
            x = table.timestamp - table.timestamp.min()
            ax.plot(x, table[key], color=group_colors[group], alpha=0.3)

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

    handles = [
        patches.Patch(
            color=group_colors[_], label=_, alpha=0.3
        )
        for _ in groups
    ]
    ax = axes[0]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1, 1.1),
        loc='lower right',
        borderaxespad=0.,
        ncol=3
    )

    fig.set_size_inches(width, height)
    fig.tight_layout()

    return fig


######
#
#   STATS
#
######


@dataclass
class BenchStats:
    input_size: int
    total_time: int
    max_gpu_ram: int


@dataclass
class TaskStats:
    task: str
    gpu_ram: int
    rps: int


def bench_stats(bench, gpu_usage_treshold=0.1):
    if not bench.records:
        raise ValueError(f'no bench records {bench.path}')

    total_time = (
        bench.records[-1].timestamp
        - bench.records[0].timestamp
    )
    max_gpu_ram = max(
        _.gpu_ram or 0
        for _ in bench.records
    )
    return BenchStats(
        bench.input_size,
        total_time,
        max_gpu_ram
    )


def task_stats(benches):
    tasks = {_.task for _ in benches}
    if len(tasks) > 1:
        raise ValueError('multiple tasks {sorted(tasks)}')

    if not any(_.input_size == 1 for _ in benches):
        raise ValueError('no input_size == 1 benches')

    if not any(_.input_size > 1 for _ in benches):
        raise ValueError('not input_size > 1 benches')

    stats = [
        bench_stats(_) for _ in benches
        if _.input_size == 1
    ]
    gpu_ram = statistics.median(_.max_gpu_ram for _ in stats)
    init_time = statistics.median(_.total_time for _ in stats)

    stats = [
        bench_stats(_) for _ in benches
        if _.input_size > 1
    ]
    rps = statistics.median(
        _.input_size / (_.total_time - init_time)
        for _ in stats
    )

    task = benches[0].task
    return TaskStats(task, gpu_ram / GB, rps)


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
        batch_size=args.batch_size,
        device=args.device,
        model_path=args.model_path,
        volume_path=args.volume_path
    )
    items = (asdict(_) for _ in records)
    print_jsonl(items)


def cli_plot(args):
    log(f'Plot {args.bench_paths!r} -> {args.image_path!r}')
    benches = [
        load_bench(_)
        for _ in args.bench_paths
    ]
    fig = plot_benches(benches)
    fig.savefig(args.image_path)


def cli_stats(args):
    log(f'Stats {args.bench_paths!r}')
    benches = [
        load_bench(_)
        for _ in args.bench_paths
    ]
    stats = task_stats(benches)
    item = asdict(stats)
    print(format_json(item))


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

    sub = subs.add_parser('bench')
    sub.set_defaults(function=cli_bench)
    sub.add_argument('image')
    sub.add_argument('data_dir', type=existing_path)
    sub.add_argument('task', choices=TASKS)
    sub.add_argument('--input-size', type=int, default=10000)
    sub.add_argument('--batch-size', type=int, default=128)
    sub.add_argument('--device', type=str, default='cuda')
    sub.add_argument('--model-path', type=str, default=None)
    sub.add_argument('--volume-path', type=str, default=None)

    sub = subs.add_parser('plot')
    sub.set_defaults(function=cli_plot)
    sub.add_argument('bench_paths', nargs='+', type=existing_path)
    sub.add_argument('image_path')

    sub = subs.add_parser('stats')
    sub.set_defaults(function=cli_stats)
    sub.add_argument('bench_paths', nargs='+', type=existing_path)

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
