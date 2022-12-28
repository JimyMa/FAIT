from typing import List

import torch

class SimpleLoop(torch.nn.Module):
    def forward(self, pred_maps: List[torch.Tensor]):
        featmap_strides = [32, 16, 8]
        num_imgs = pred_maps[0].shape[0]

        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 85)
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)
            flatten_strides.append(torch.tensor(stride).expand(pred.size(1)))
        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_strides = torch.cat(flatten_strides)

        return flatten_preds, flatten_strides

mod = torch.jit.script(SimpleLoop())
mod.eval()
mod = torch.jit.freeze(mod)
print(mod.graph)
# torch.jit.save(mod, 'simple_loop.pt')
