import ast
import inspect
from argparse import ArgumentParser, Namespace
from typing import Dict

import torch
import torch._dynamo

from prof import (ProfileRewriter, fmt_duration, print_profiling_results,
                  prof_begin, prof_end)
from run_utils import evaluate, to_cuda

args = Namespace()


module_cls_names: Dict[str, str] = {
    'yolov3_bbox_prof': 'YOLOV3BBox',
    'yolact_mask_prof': 'YolactBBoxMask',
}


def parse_args():
    global args
    parser = ArgumentParser()
    parser.add_argument('-m', '--module', type=str,
                        choices=module_cls_names.keys(),
                        help='Python module name under `models`.')
    parser.add_argument('-f', '--feature', type=str,
                        help='Pickle file of network output features.')
    parser.add_argument('-c', '--compile', type=str,
                        help='Compile the module with specific backend.')
    args = parser.parse_args()


if __name__ == '__main__':
    parse_args()

    if args.compile:
        torch._dynamo.reset()
        torch._dynamo.config.suppress_errors = True

    mod_name = module_cls_names[args.module]
    feats = torch.load(args.feature)
    num_samples = len(feats)

    exec(f'from models.{args.module} import *')
    src = ast.parse(inspect.getsource(eval(mod_name)))
    src = ProfileRewriter().visit(src)
    code = compile(src, '<string>', 'exec')
    exec(code)
    mod = eval(mod_name)().eval().cuda()
    if args.compile:
        mod = torch.compile(mod, backend=args.compile, dynamic=True)

    def task(idx: int):
        mod(*to_cuda(feats[idx % num_samples]))

    for i in range(num_samples):
        task(i)

    result = evaluate(task)
    print(f'latency: {fmt_duration(result.mean())}')
    print(f'count: {result.count}')

    print_profiling_results()
