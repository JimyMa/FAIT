from typing import List

import torch


class LoopAssign(torch.nn.Module):
    def forward(self, a_list: List[torch.Tensor], b_list: List[float]):
        c_list = []
        for a, b in zip(a_list, b_list):
            a = a.permute(0, 2, 3, 1)
            a[..., :2].sigmoid_()
            c = a + b
            c_list.append(c)

        return c_list


mod = torch.jit.script(LoopAssign())
mod.eval()
mod = torch.jit.freeze(mod)
print(mod.graph)
torch.jit.save(mod, 'loop_assign.pt')
