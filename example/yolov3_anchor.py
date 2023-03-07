from typing import List, Tuple

import torch


class YOLOAnchorGenerator(torch.nn.Module):
    def __init__(self, strides, base_sizes):
        super().__init__()
        from torch.nn.modules.utils import _pair
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.base_sizes = []
        for base_sizes_per_level in base_sizes:
            # assert len(base_sizes[0]) == len(base_sizes_per_level)
            self.base_sizes.append(
                [_pair(base_size) for base_size in base_sizes_per_level])
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_levels(self):
        return len(self.base_sizes)

    def gen_base_anchors(self):
        multi_level_base_anchors: List[torch.Tensor] = []
        for i, base_sizes_per_level in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_sizes_per_level,
                                                   center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, base_sizes_per_level: List[Tuple[int, int]], center: Tuple[float]):
        x_center, y_center = center
        base_anchors = []
        for base_size in base_sizes_per_level:
            w, h = base_size
            base_anchor = torch.Tensor([
                x_center - 0.5 * w, y_center - 0.5 * h, x_center + 0.5 * w,
                y_center + 0.5 * h
            ])
            base_anchors.append(base_anchor)
        base_anchors = torch.stack(base_anchors, dim=0)
        return base_anchors.cuda()

    def forward(self, featmap_sizes: List[Tuple[int, int]]):
        # assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=torch.float32, device=torch.device('cuda'))
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int, int],
                                 level_idx: int,
                                 dtype: torch.dtype,
                                 device: torch.device):
        base_anchors = self.base_anchors[level_idx].to(device, dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = torch.arange(0, feat_w, dtype=dtype,
                               device=device) * stride_w
        shift_y = torch.arange(0, feat_h, dtype=dtype,
                               device=device) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)

        return all_anchors

    def _meshgrid(self, x: torch.Tensor, y: torch.Tensor):
        xx = x.repeat(y.size(0))
        yy = y.view(-1, 1).repeat(1, x.size(0)).view(-1)
        return xx, yy


class YOLOV3BBox(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor_generator = YOLOAnchorGenerator(
            strides=[32, 16, 8],
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]]
        )

    def forward(self, pred_maps: List[torch.Tensor]):
        featmap_sizes = [(pred_map.size(-2), pred_map.size(-1))
                         for pred_map in pred_maps]
        return self.anchor_generator(featmap_sizes)


if __name__ == '__main__':
    mod = YOLOV3BBox().cuda().eval()
    mod = torch.jit.script(mod)
    print(mod.graph)
    torch.jit.save(mod, 'yolov3_anchor.pt')
