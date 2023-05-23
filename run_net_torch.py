from argparse import ArgumentParser, Namespace
from typing import Dict, Type

import torch
import torch._dynamo

from models.fcos_net import FCOS
from models.solov2_net import SOLOV2
from models.ssd_net import SSD
from models.yolact_net import Yolact
from models.yolov3_net import YOLOV3
from prof import fmt_duration
from run_utils import evaluate

args = Namespace()

module_classes: Dict[str, Type[torch.nn.Module]] = {
    'yolov3': YOLOV3,
    'ssd': SSD,
    'fcos': FCOS,
    'yolact': Yolact,
    'solov2': SOLOV2,
}


def parse_args():
    global args
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str,
                        choices=module_classes.keys(),
                        help='Model name.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Pickle file of COCO128 dataset.')
    parser.add_argument('-t', '--trace', action='store_true',
                        help='Whether to trace the model or not')
    parser.add_argument('-c', '--compile', type=str, 
                        help='Compile the model with specific backend.')
    args = parser.parse_args()


def main():
    if args.compile:
        torch._dynamo.reset()
        torch._dynamo.config.suppress_errors = True

    mod = module_classes[args.model]().cuda().eval()
    params = torch.load(mod.ckpt_file)
    if 'state_dict' in params.keys():
        params = params['state_dict']
    dataset = torch.load(args.dataset).cuda()
    num_samples = len(dataset)
    if args.trace:
        mod = torch.jit.trace(mod, dataset[0:1])
    elif args.compile:
        mod = torch.compile(mod, backend=args.compile)

    def task(idx: int):
        idx %= num_samples
        mod(dataset[idx:idx+1])

    print(f'Latency: {fmt_duration(evaluate(task).mean())}')


if __name__ == '__main__':
    parse_args()
    main()
