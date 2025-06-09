import torch
import torch.nn as nn

import iree.turbine.aot as aot
import iree.runtime as rt


class Attention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.softmax(q @ k.T, dim=-1) @ v


model = Attention()

seq_len = 1024
head_dim = 128

q = torch.zeros((seq_len, head_dim), dtype=torch.float16)
k = torch.zeros((seq_len, head_dim), dtype=torch.float16)
v = torch.zeros((seq_len, head_dim), dtype=torch.float16)

exported = aot.export(model, args=(q, k, v))

exported.print_readable()
