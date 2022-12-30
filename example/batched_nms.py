import torch
import torchvision


class BatchedNMS(torch.nn.Module):
    def forward(self, boxes: torch.Tensor,
                scores: torch.Tensor,
                idxs: torch.Tensor):
        iou_threshold = 0.45

        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (
            max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

        total_mask = torch.zeros_like(scores, dtype=torch.bool)
        scores_after_nms = torch.zeros_like(scores)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero().view(-1)
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


mod = torch.jit.script(BatchedNMS())
mod.eval()
mod = torch.jit.freeze(mod)
print(mod.graph)
torch.jit.save(mod, 'batched_nms.pt')
