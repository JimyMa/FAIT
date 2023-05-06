from argparse import ArgumentParser, Namespace
from typing import Dict, Type

import torch
import torch._dynamo

from models.fcos_bbox import FCOSBBox
from models.solov2_mask import SOLOV2Mask
from models.ssd_bbox import SSDBBox
from models.yolact_mask import YolactBBoxMask
from models.yolov3_bbox import YOLOV3BBox
from run_utils import evaluate, fmt_duration, to_cuda

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


def main():
    torch._dynamo.config.suppress_errors = True
    mod = module_classes[args.model]().cuda().eval()
    mod = torch.compile(mod, dynamic=True)
    feats = torch.load(args.feature)
    num_samples = len(feats)

    def task(idx: int):
        mod(*to_cuda(feats[idx % num_samples]))

    print(f'torch latency: {fmt_duration(evaluate(task))}')


if __name__ == '__main__':
    parse_args()
    main()
