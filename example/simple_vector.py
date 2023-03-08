from typing import List

import torch


class SimpleVector(torch.nn.Module):
    def forward(self, a: torch.Tensor):
        a[0, 40::, 0, 0].sigmoid_()
        return a + a * a - a


mod = torch.jit.script(SimpleVector())
mod.eval()
mod = torch.jit.freeze(mod)
print(mod.graph)
torch.jit.save(mod, 'simple_vector.pt')
