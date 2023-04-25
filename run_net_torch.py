from argparse import ArgumentParser, Namespace
from time import perf_counter
from typing import Dict, Type

import torch

from models.solov2_net import SOLOV2
from models.ssd_net import SSD
from models.yolact_net import Yolact
from models.yolov3_net import YOLOV3

args = Namespace()


def parse_args():
    global args
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str,
                        choices=['yolov3', 'ssd', 'yolact', 'solov2'],
                        help='Model name.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Pickle file of COCO128 dataset.')
    args = parser.parse_args()


module_classes: Dict[str, Type[torch.nn.Module]] = {
    'yolov3': YOLOV3,
    'ssd': SSD,
    'yolact': Yolact,
    'solov2': SOLOV2,
}

warmup_runs = 16
run_duration = 2.


def main():
    mod = module_classes[args.model]().cuda().eval()
    params = torch.load(mod.ckpt_file)
    if 'state_dict' in params.keys():
        params = params['state_dict']
    dataset = torch.load(args.dataset).cuda()
    num_samples = len(dataset)
    for i in range(16):
        mod(dataset[i:i+1])
    torch.cuda.synchronize()
    count = 0
    begin = perf_counter()
    while perf_counter() - begin < run_duration:
        i = count % num_samples
        mod(dataset[i:i+1])
        torch.cuda.synchronize()
        count += 1
    print('torch latency: {:.3f}ms'.format(
        (perf_counter() - begin) / count * 1e3))


if __name__ == '__main__':
    parse_args()
    main()
