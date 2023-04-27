from argparse import ArgumentParser, Namespace
from time import perf_counter
from typing import Dict, Type

import torch

from models.fcos_bbox import FCOSBBox
from models.solov2_mask import SOLOV2Mask
from models.ssd_bbox import SSDBBox
from models.yolact_mask import YolactBBoxMask
from models.yolov3_bbox import YOLOV3BBox

args = Namespace()


module_classes: Dict[str, Type[torch.nn.Module]] = {
    'yolov3': YOLOV3BBox,
    'ssd': SSDBBox,
    'fcos': FCOSBBox,
    'yolact': YolactBBoxMask,
    'solov2': SOLOV2Mask,
}


def parse_args():
    global args
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str,
                        choices=module_classes.keys(),
                        help='Model name.')
    parser.add_argument('-f', '--feature', type=str,
                        help='Pickle file of network output features.')
    args = parser.parse_args()


warmup_runs = 16
run_duration = 2.


def to_cuda(val):
    if isinstance(val, list):
        return [to_cuda(elem) for elem in val]
    elif isinstance(val, tuple):
        return tuple(to_cuda(elem) for elem in val)
    elif isinstance(val, torch.Tensor):
        return val.cuda()
    else:
        return val


def main():
    mod = module_classes[args.model]().cuda().eval()
    feats = to_cuda(torch.load(args.feature))
    num_samples = len(feats)
    for i in range(16):
        mod(*feats[i])
    torch.cuda.synchronize()
    count = 0
    begin = perf_counter()
    while perf_counter() - begin < run_duration:
        mod(*feats[count % num_samples])
        torch.cuda.synchronize()
        count += 1
    print('torch latency: {:.3f}ms'.format(
        (perf_counter() - begin) / count * 1e3))


if __name__ == '__main__':
    parse_args()
    main()
