from time import perf_counter
from typing import Callable

import torch

from prof import fmt_duration, enable_profiling, disable_profiling


def to_cuda(val):
    if isinstance(val, list):
        return [to_cuda(elem) for elem in val]
    elif isinstance(val, tuple):
        return tuple(to_cuda(elem) for elem in val)
    elif isinstance(val, torch.Tensor):
        return val.cuda()
    else:
        return val


warmup_runs = 16
run_duration = 2.


def evaluate(task: Callable[[int], None]):
    disable_profiling()
    for i in range(warmup_runs):
        task(i)

    enable_profiling()
    torch.cuda.synchronize()
    count = 0
    begin = perf_counter()
    while perf_counter() - begin < run_duration:
        task(count)
        torch.cuda.synchronize()
        count += 1

    return (perf_counter() - begin) / count
