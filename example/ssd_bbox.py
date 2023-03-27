from typing import List, Tuple

import numpy as np
import torch
import torchvision


class SSDAnchorGenerator(torch.nn.Module):
    def __init__(self,
                 strides: List[int],
                 ratios: List[List[int]],
                 min_sizes: List[float],
                 max_sizes: List[float],
                 scale_major: bool):
        # assert len(strides) == len(ratios)
        # assert len(min_sizes) == len(max_sizes) == len(strides)
        super().__init__()
        from torch.nn.modules.utils import _pair
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]

        anchor_ratios = []
        anchor_scales = []
        for k in range(len(self.strides)):
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            anchor_ratio = [1.]
            for r in ratios[k]:
                anchor_ratio += [1 / r, r]  # 4 or 6 ratio
            anchor_ratios.append(torch.Tensor(anchor_ratio).cuda())
            anchor_scales.append(torch.Tensor(scales).cuda())

        self.base_sizes = min_sizes
        self.scales = anchor_scales
        self.ratios = anchor_ratios
        self.scale_major = scale_major
        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        multi_level_base_anchors: List[torch.Tensor] = []
        for i, base_size in enumerate(self.base_sizes):
            base_anchors = self.gen_single_level_base_anchors(
                base_size,
                scales=self.scales[i],
                ratios=self.ratios[i],
                center=self.centers[i])
            indices = list(range(len(self.ratios[i])))
            indices.insert(1, len(indices))
            base_anchors = torch.index_select(base_anchors, 0,
                                              torch.LongTensor(indices))
            multi_level_base_anchors.append(base_anchors.cuda())
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center):
        w = base_size
        h = base_size
        x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    @property
    def num_levels(self):
        return len(self.strides)

    def forward(self, featmap_sizes: List[Tuple[int, int]], dtype: torch.dtype, device: torch.device):
        # assert self.num_levels == len(featmap_sizes)
        multi_level_anchors: List[torch.Tensor] = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=dtype, device=device)
            # print(anchors.shape)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(self,
                                 featmap_size: Tuple[int, int],
                                 level_idx: int,
                                 dtype: torch.dtype,
                                 device: torch.device):
        base_anchors = self.base_anchors[level_idx].to(
            device=device, dtype=dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = torch.arange(0, feat_w, device=device,
                               dtype=dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device,
                               dtype=dtype) * stride_h
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        return all_anchors

    def _meshgrid(self, x: torch.Tensor, y: torch.Tensor):
        xx = x.repeat(y.size(0))
        yy = y.view(-1, 1).repeat(1, x.size(0)).view(-1)
        return xx, yy


def delta2bbox(rois: torch.Tensor,
               deltas: torch.Tensor
               ):
    means = (0., 0., 0., 0.)
    stds = (0.1, 0.1, 0.2, 0.2)
    wh_ratio_clip = 0.016
    max_shape = (320, 320)

    num_classes = deltas.size(1) // 4
    deltas = deltas.reshape(-1, 4)
    means = torch.tensor(means, dtype=deltas.dtype,
                         device=deltas.device).view(1, -1)
    stds = torch.tensor(stds, dtype=deltas.dtype,
                        device=deltas.device).view(1, -1)
    denorm_deltas = deltas * stds + means
    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
    pwh = (rois_[:, 2:] - rois_[:, :2])
    dxy_wh = pwh * dxy
    max_ratio = float(torch.abs(torch.log(torch.tensor(wh_ratio_clip))))
    dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    x1y1 = gxy - (gwh * 0.5)
    x2y2 = gxy + (gwh * 0.5)
    bboxes = torch.cat([x1y1, x2y2], dim=-1)
    bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
    bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])

    return bboxes


def select_single_mlvl(mlvl_tensors: List[torch.Tensor], batch_id: int):
    num_levels = len(mlvl_tensors)
    mlvl_tensor_list = [mlvl_tensors[i][batch_id] for i in range(num_levels)]
    return mlvl_tensor_list


def filter_scores_and_topk(scores: torch.Tensor, score_thr: float, topk: int):
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)
    num_topk = min(topk, valid_idxs.size(0))
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs[:, 0], topk_idxs[:, 1]
    return scores, labels, keep_idxs


def batched_nms(boxes: torch.Tensor,
                scores: torch.Tensor,
                idxs: torch.Tensor):
    iou_threshold = 0.45
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    split_thr = 10000
    if boxes_for_nms.size(0) < split_thr:
        dets, keep = nms_wrapper(
            boxes_for_nms, scores, iou_threshold=iou_threshold)
        boxes = boxes[keep]
        scores = dets[:, -1]
    else:
        total_mask = torch.zeros_like(scores, dtype=torch.bool)
        scores_after_nms = torch.zeros_like(scores)
        for id in torch.unique(idxs):
            mask = torch.nonzero(idxs == id).view(-1)
            dets, keep = nms_wrapper(
                boxes_for_nms[mask], scores[mask], iou_threshold)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero().view(-1)
        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


def nms_wrapper(boxes: torch.Tensor,
                scores: torch.Tensor,
                iou_threshold: float):
    # assert boxes.size(1) == 4
    # assert boxes.size(0) == scores.size(0)
    inds = torchvision.ops.nms(boxes, scores, iou_threshold)
    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    return dets, inds


class SSDBBox(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_classes = 80
        self.cls_out_channels = self.num_classes + 1
        self.use_sigmoid_cls = False
        self.prior_generator = SSDAnchorGenerator(
            strides=[16, 32, 64, 107, 160, 320],
            ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
            min_sizes=[48, 100, 150, 202, 253, 304],
            max_sizes=[100, 150, 202, 253, 304, 320],
            scale_major=False
        )

    def forward(self,
                cls_scores: List[torch.Tensor],
                bbox_preds: List[torch.Tensor]):
        # cls_scores: [torch.Size([1, 486, 20, 20]), torch.Size([1, 486, 10, 10]), torch.Size([1, 486, 5, 5]), torch.Size([1, 486, 3, 3]), torch.Size([1, 486, 2, 2]), torch.Size([1, 486, 1, 1])]
        # bbox_preds: [torch.Size([1, 24, 20, 20]), torch.Size([1, 24, 10, 10]), torch.Size([1, 24, 5, 5]), torch.Size([1, 24, 3, 3]), torch.Size([1, 24, 2, 2]), torch.Size([1, 24, 1, 1])]
        # assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [(cls_score.size(-2), cls_score.size(-1))
                         for cls_score in cls_scores]
        mlvl_priors = self.prior_generator(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for img_id in range(cls_scores[0].size(0)):
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            print([a.shape for a in cls_score_list])
            print([a.shape for a in bbox_pred_list])
            print([a.shape for a in mlvl_priors])
            results = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, mlvl_priors)
            result_list.append(results)

        return result_list

    def _get_bboxes_single(self,
                           cls_score_list: List[torch.Tensor],
                           bbox_pred_list: List[torch.Tensor],
                           mlvl_priors: List[torch.Tensor]):
        nms_pre = 1000
        score_thr = 0.02

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, priors in zip(cls_score_list, bbox_pred_list, mlvl_priors):
            # assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cls_score = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = cls_score.softmax(-1)[:, :-1]
            scores, labels, keep_idxs = filter_scores_and_topk(
                scores, score_thr, nms_pre)

            bbox_pred = bbox_pred[keep_idxs]
            priors = priors[keep_idxs]
            bboxes = delta2bbox(priors, bbox_pred)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        max_per_img = 100
        det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                            mlvl_labels)
        det_bboxes = det_bboxes[:max_per_img]
        det_labels = mlvl_labels[keep_idxs][:max_per_img]

        return det_bboxes, det_labels


if __name__ == '__main__':
    mod = SSDBBox().cuda().eval()
    mod = torch.jit.script(mod)
    print(mod.graph)
    torch.jit.save(mod, 'ssd_bbox.pt')
