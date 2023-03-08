from typing import List

import torch


class SimpleLoop(torch.nn.Module):
    def forward(self, a_list: List[torch.Tensor], b_list: List[float]):
        c_list = []
        for a, b in zip(a_list, b_list):
            tmp = 2.0 * a / 2.0 + b
            c = tmp + a + b
            c_list.append(c)

        return c_list


mod = torch.jit.script(SimpleLoop())
mod.eval()
mod = torch.jit.freeze(mod)
print(mod.graph)
torch.jit.save(mod, 'loop_vector_add.pt')
