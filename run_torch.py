from argparse import ArgumentParser, Namespace
from time import perf_counter
from typing import Dict, Type

import torch

from models.solov2_mask import SOLOV2Mask
from models.ssd_bbox import SSDBBox
from models.yolact_mask import YolactBBoxMask
from models.yolov3_bbox import YOLOV3BBox

args = Namespace()


def parse_args():
    global args
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str,
                        choices=['yolov3', 'ssd', 'yolact', 'solov2'],
                        help='Model name.')
    parser.add_argument('-p', '--prediction', type=str,
                        help='Pickle file of model predictions.')
    args = parser.parse_args()


module_classes: Dict[str, Type[torch.nn.Module]] = {
    'yolov3': YOLOV3BBox,
    'ssd': SSDBBox,
    'yolact': YolactBBoxMask,
    'solov2': SOLOV2Mask,
}

warmup_runs = 16
run_duration = 2.


def main():
    mod = module_classes[args.model]().cuda().eval()
    preds = torch.load(args.prediction)
    num_samples = len(preds)
    for i in range(16):
        mod(*preds[i])
    torch.cuda.synchronize()
    count = 0
    begin = perf_counter()
    while perf_counter() - begin < run_duration:
        mod(*preds[count % num_samples])
        torch.cuda.synchronize()
        count += 1
    print('torch latency: {:.3f}ms'.format(
        (perf_counter() - begin) / count * 1e3))


if __name__ == '__main__':
    parse_args()
    main()
