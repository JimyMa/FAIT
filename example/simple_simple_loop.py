from typing import List

import torch


class SimpleSimpleLoop(torch.nn.Module):
    def forward(self, pred_maps: List[torch.Tensor]):
        featmap_strides = [32, 16, 8]
        flatten_preds = []
        for pred, stride in zip(pred_maps, featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape([1, -1, 85])
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)

        return flatten_preds


mod = torch.jit.script(SimpleSimpleLoop())
mod.eval()
mod = torch.jit.freeze(mod)
print(mod.graph)
torch.jit.save(mod, 'simple_simple_loop.pt')
