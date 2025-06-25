import math
import triton
import triton.language as tl
import torch
import torch.nn as nn

import iree.turbine.aot as aot
import iree.runtime as rt

# SRAM size
M = int(262144)

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit 
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return tl.softmax(q @ k.T) @ v

@triton.jit
def flash_attention_kernel( q_ptr: torch.Tensor, 
                            k_ptr: torch.Tensor, 
                            v_ptr: torch.Tensor,
                            o_ptr: torch.Tensor,
                            N: int,
                            d: int,
                            B_c: int, 
                            B_r: int,
                            ) -> torch.Tensor:
    pid = tl.program_id(axis=0)
    T_r = tl.cdiv(N, B_r)
    T_c = tl.cdiv(N, B_c)

    # This code is as per Algorithm 1 in Flash Attention 2: https://arxiv.org/pdf/2307.08691
    ##  3: for 1 <= i <= T_r do
    for i in tl.range(0, N, B_r):
        ## 4: load Q_i from HBM to on-chip SRAM
        off_r = tl.arange(0, B_r)        # off_r has shape [B_r]
        q_i = tl.load(q_ptr + offsets)   # q_i   has shape [B_r, d]
        o_i = tl.load(o_ptr + offsets)   # o_i   has shape [B_r, d]

        ## 5: on-chip, initialize o_prev_i, l_prev_i, m_prev_i
        o_prev = tl.zeros((B_r, d), dtype=torch.float16)             # 5: o(0)_i has 0s    of shape [B_r, d]
        l_prev = tl.zeros((B_r),    dtype=torch.float16)             # 5: l(0)_i has 0s    of shape [B_r]
        m_prev = tl.zeros((B_r, 1), dtype=torch.float16) - math.inf  # 5: m(0)_i has -infs of shape [B_r]

        # online softmax
        ## 6: for 1 <= j <= T_c do
        for j in tl.range(0, N, B_c):
            k_j = k_blocks[j]  ## 7: load K_j from HBM to on-chip SRAM. K_j has shape [B_c, d]
            v_j = v_blocks[j]  ## 7: load V_j from HBM to on-chip SRAM. V_j has shape [B_c, d]

            s_i = q_i @ k_j.T                                             ## 8: On chip, compute S(j)_i = ${ Q_i @ K_j.T                                        }$ of shape [B_r, B_c]
            m_tilde, _ = tl.max(s_i, dim=-1, keepdim=True)                ## 9: On chip, compute m(j)_i = ${ max( m(j-1)_i, rowmax(S(j)_i) )                    }$ of shape [B_r]
            m_i = tl.maximum(m_prev, m_tilde)                                                
            p_i = tl.exp(s_i - m_i)                                       ## 9: On chip, compute P(j)_i = ${ exp( S(j)_i - m(j)_i )                             }$ of shape [B_r, B_c]
            ex_m_diff = tl.exp(m_prev - m_i)
            l_i = ex_m_diff * l_prev + tl.sum(p_i, dim=-1, keepdim=True)  ## 9: On chip, compute l(j)_i = ${ exp( m(j-1)_i - m(j)_i ) * l(j)_i + rowsum(P(j)_i) }$ of shape [B_r]
            o_i = ex_m_diff * o_prev + p_i @ v_j                          ## 10: On chip, compute O(j)_i = ${ diag(exp( m(j-1)_i - m(j)_i ) + P(j)_i @ V_j )    }$ of shape [B_r, d]

            m_prev = m_i
            l_prev = l_i
            o_prev = o_i

        tl.store(o + offsets, o_i / l_prev)
        tl.store(l + offsets, m_prev + tl.log(l_prev))

    return q

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    N,d = q.shape
    assert N == k.shape[0] and d == k.shape[1]
    o = tl.zeros((N,d), dtype=q.dtype)
    
    return o

def main():
    seq_len = 1024
    head_dim = 128

    q = torch.randn((seq_len, head_dim), device=DEVICE, dtype=torch.float16)
    k = torch.randn((seq_len, head_dim), device=DEVICE, dtype=torch.float16)
    v = torch.randn((seq_len, head_dim), device=DEVICE, dtype=torch.float16)

    flash_output = flash_attention(q,k,v)
    base_output = attention(q,k,v)
    rtol = 0
    if torch.allclose(flash_output, base_output, atol=1e-2, rtol=rtol):
        print("✅ Flash and Base match")
    else:
        print("❌ Flash and Base differ")