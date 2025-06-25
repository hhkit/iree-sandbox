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
def flash_attention_kernel( q: torch.Tensor, 
                            k: torch.Tensor, 
                            v: torch.Tensor,
                            o: torch.Tensor,
                            o_prev: torch.Tensor,
                            l_prev: torch.Tensor,
                            m_prev: torch.Tensor,
                            ) -> torch.Tensor:
    N,d = q.shape
    B_c = int(math.ceil(float(M) / (4 * d)))
    B_r = int(min(B_c, d))
    T_r = int(math.ceil(float(N) / B_r))
    T_c = int(math.ceil(float(N) / B_c))

    pid = tl.program_id(axis=0)
    
    for i in tl.range(0, N, B_r):
        offsets = tl.arange(0, B_r)
        q_i = tl.load(q + offsets)
        o_i = tl.load(o + offsets)

        o_prev = torch.zeros((B_r, d), dtype=torch.float16)
        l_prev = tl.zeros((B_r), dtype=torch.float16)
        m_prev = tl.zeros((B_r, 1), -math.inf, dtype=torch.float16) - math.inf

        # online softmax
        for j in tl.range(0, N, B_c):
            k_j = k_blocks[j]
            v_j = v_blocks[j]

            s_i = q_i @ k_j.T
            m_tilde, _ = tl.max(s_i, dim=-1, keepdim=True)
            m_i = tl.maximum(m_prev, m_tilde)
            p_i = tl.exp(s_i - m_i)
            ex_m_diff = tl.exp(m_prev - m_i)
            l_i = ex_m_diff * l_prev + tl.sum(p_i, dim=-1, keepdim=True)
            o_i = ex_m_diff * o_prev + p_i @ v_j

            m_prev = m_i
            l_prev = l_i
            o_prev = o_i

        tl.store(o + offsets, o_i / l_prev)
        tl.store(l + offsets, m_prev + tl.log(l_prev))

    return q

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    o = tl.zeros(q.shape, dtype=q.dtype)
    o_prev = torch.zeros((B_r, d), dtype=torch.float16)
    l_prev = torch.zeros((B_r), dtype=torch.float16)
    m_prev = torch.full((B_r, 1), -math.inf, dtype=torch.float16)
    return o

def main():
    seq_len = 1024
    head_dim = 128

    q = tl.zeros((seq_len, head_dim), dtype=torch.float16)
    k = tl.zeros((seq_len, head_dim), dtype=torch.float16)
    v = tl.zeros((seq_len, head_dim), dtype=torch.float16)

    attention(q, k, v)

    exported = aot.export(attention, args=(q, k, v))

    exported.print_readable()
