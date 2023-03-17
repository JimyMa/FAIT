import torch


class DecodeBBoxes(torch.nn.Module):
    def forward(self, bboxes, pred_bboxes, stride):
        xy_centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5 + (
            pred_bboxes[..., :2] - 0.5) * stride
        whs = (bboxes[..., 2:] -
               bboxes[..., :2]) * 0.5 * pred_bboxes[..., 2:].exp()
        decoded_bboxes = torch.stack(
            (xy_centers[..., 0] - whs[..., 0], xy_centers[..., 1] -
                whs[..., 1], xy_centers[..., 0] + whs[..., 0],
                xy_centers[..., 1] + whs[..., 1]),
            dim=-1)
        return decoded_bboxes


if __name__ == '__main__':
    mod = DecodeBBoxes().cuda().eval()
    mod = torch.jit.script(mod)
    # mod = torch.jit.freeze(mod)
    print(mod.graph)
    torch.jit.save(mod, 'decode_bboxes.pt')
