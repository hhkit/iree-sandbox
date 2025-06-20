import math
import triton
import triton.language as tl
import torch
import torch.nn as nn

import iree.turbine.aot as aot
import iree.runtime as rt

# SRAM size
M = int(262144)

@triton.jit 
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) ->  torch.Tensor:
    return tl.softmax(q @ k.T) @ v

@triton.jit
def flashattention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    N,d = q.shape
    B_c = int(math.ceil(float(M) / (4 * d)))
    B_r = int(min(B_c, d))
    T_r = int(math.ceil(float(N) / B_r))
    T_c = int(math.ceil(float(N) / B_c))

    pid = tl.program_id(axis=0)

    o = tl.zeros(q.shape, dtype=tl.float16)
    l = tl.zeros((N), dtype=tl.float16)
    m = tl.full((N,), -math.inf, dtype=tl.float16)
    
    for j in tl.range(0, N, B_c):
        k_j = tl.load(k[j : j + B_c, :]).T
        v_j = tl.load(v[j : j + B_c, :])

        for i in tl.range(0, N, B_r):
            q_i = tl.load(q[i : i + B_r, :])
            o_i = tl.load(o[i : i + B_r, :])
            l_i = tl.load(l[i : i + B_r, :])
            m_i = tl.load(m[i : i + B_r, :])

            s = q_i @ k_j
            m_ij_tilde, _ = s_block.max(dim=-1, keepdim=True)
            p_ij_tilde = torch.exp(s_block - m_ij_tilde)
            l_ij_tilde = torch.sum(p_ij_tilde, dim=-1, keepdim=True)

            m_i_new = m_i.max(m_ij_tilde)
            m_sub_mnew = torch.exp(m_i - m_i_new)
            m_tilde_sub_mnew = torch.exp(m_ij_tilde - m_i_new)

            l_i_new = m_sub_mnew * l_i + m_tilde_sub_mnew * l_ij_tilde

            lexpr = ((l_i / l_i_new) * torch.exp(m_i - m_i_new) @ o_i)
            rexpr = (torch.exp(m_ij_tilde - m_i_new) /
                        l_i_new) @ (p_ij_tilde @ v_j)
            o_blocks[i] = lexpr + rexpr
            l_blocks[i] = l_i_new
            m_blocks[i] = m_i_new

            tl.store()

    return q

@triton.jit
def main():
    seq_len = 1024
    head_dim = 128

    q = tl.zeros((seq_len, head_dim), dtype=torch.float16)
    k = tl.zeros((seq_len, head_dim), dtype=torch.float16)
    v = tl.zeros((seq_len, head_dim), dtype=torch.float16)

    attention(q, k, v)

    exported = aot.export(attention, args=(q, k, v))

    exported.print_readable()
