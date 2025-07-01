import torch
import torch.nn as nn

import iree.turbine.aot as aot
import iree.runtime as rt


class Attention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # return torch.nn.functional.scaled_dot_product_attention(q, k ,v) # this is flash attention
        s = q @ k.T
        m, _ = torch.max(s, dim=-1, keepdim=True)
        sum = torch.sum(torch.exp(m - s))
        acc = s @ v
        return acc / sum
        return torch.softmax(s, dim=-1) @ v                     # this is attention


model = Attention()

seq_len = 1024
head_dim = 128

q = torch.zeros((seq_len, head_dim), dtype=torch.float16)
k = torch.zeros((seq_len, head_dim), dtype=torch.float16)
v = torch.zeros((seq_len, head_dim), dtype=torch.float16)

exported = aot.export(model, args=(q, k, v))

exported.print_readable()
