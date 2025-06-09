import math
import torch
import torch.nn as nn

import iree.turbine.aot as aot
import iree.runtime as rt

# SRAM size
M = 512


class FlashAttention(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        N = q.shape[0]
        d = float(q.shape[-1])
        B_c = math.ceil(M / (4 * d))
        B_r = min(B_c, d)

        o = torch.zeros(q.shape, dtype=torch.float16)
        l = torch.zeros((N), dtype=torch.float16)
        m = torch.fill(torch.empty((N), dtype=torch.float16), -math.inf)

        T_r = math.ceil(N / B_r)
        T_c = math.ceil(N / B_c)

        q_blocks = q.split(B_r, dim=0)
        k_blocks = k.split(B_c, dim=0)
        v_blocks = v.split(B_c, dim=0)

        o_blocks = list(o.split(B_r, dim=0))
        l_blocks = list(l.split(B_r, dim=0))
        m_blocks = list(m.split(B_r, dim=0))

        # Source: FlashAttention (doi/2205.14135) Algorithm 1

        # for (k_i, v_i) in zip(k_blocks, v_blocks):
        for j in range(T_c):
            k_j = k_blocks[j]
            v_j = v_blocks[j]

            # for (q_i, o_i, l_i, m_i) in zip(q_blocks, o_blocks, l_blocks, m_blocks):
            for i in range(T_r):
                print(f"{i}x{j}")

                q_i = q_blocks[i]
                o_i = o_blocks[i]
                l_i = l_blocks[i]
                m_i = m_blocks[i]

                s_block = q_i @ k_j.T

                # todo: broadcast m_ij to match s_block
                m_ij_tilde, _ = s_block.max(dim=-1, keepdim=True)
                p_ij_tilde = torch.exp(s_block - m_ij_tilde)
                l_ij_tilde = torch.sum(p_ij_tilde, dim=-1, keepdim=True)

                m_i_new = m_i.max(m_ij_tilde)
                m_sub_mnew = torch.exp(m_i - m_i_new)
                m_tilde_sub_mnew = torch.exp(m_ij_tilde - m_i_new)

                l_i_new = m_sub_mnew * l_i + m_tilde_sub_mnew * l_ij_tilde

                o_blocks[i] = (((l_i) * torch.exp(m_i - m_i_new) @ o_i) +
                               (torch.exp(m_ij_tilde - m_i_new)) @ (p_ij_tilde @ v_j)) / l_i_new
                l_blocks[i] = l_i_new
                m_blocks[i] = m_i_new

        return torch.cat(o_blocks)


model = FlashAttention()

seq_len = 1024
head_dim = 128

q = torch.zeros((seq_len, head_dim), dtype=torch.float16)
k = torch.zeros((seq_len, head_dim), dtype=torch.float16)
v = torch.zeros((seq_len, head_dim), dtype=torch.float16)

exported = aot.export(model, args=(q, k, v))

exported.print_readable()
