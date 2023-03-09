
from typing import List

import torch


class Normalize(torch.nn.Module):
    def forward(self, src:torch.Tensor):
        # RGB to BGR
        dup = src + src
        dup[:, :, 0] = src[:, :, 2]
        dup[:, :, 2] = src[:, :, 0]
        return (dup - 0.0) * 1.0


mod = torch.jit.script(Normalize())
mod.eval()
mod = torch.jit.freeze(mod)
print(mod.graph)
torch.jit.save(mod, 'normalize.pt')
