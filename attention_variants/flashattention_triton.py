import math
import triton
import triton.language as tl
import torch
import torch.nn as nn
import numpy 
import pandas

import iree.turbine.aot as aot
import iree.runtime as rt

# SRAM size
M = int(262144)

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def torch_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q,k,v)

def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.softmax(q @ k.T, dim=-1) @ v

def get_autotune_config():
    return [
        triton.Config({'B_c': 16, 'B_r': 16, 'd': 128}),
        triton.Config({'B_c': 16, 'B_r': 32, 'd': 128}),
        triton.Config({'B_c': 32, 'B_r': 16, 'd': 128}),
        triton.Config({'B_c': 32, 'B_r': 32, 'd': 128}),
    ]

@triton.jit 
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return tl.softmax(q @ k.T) @ v

@triton.autotune(
    configs=get_autotune_config(),
    key=['B_c', 'B_r', 'd'],
)
@triton.jit
def flash_attention_kernel( q_ptr: torch.Tensor, 
                            k_ptr: torch.Tensor, 
                            v_ptr: torch.Tensor,
                            o_ptr: torch.Tensor,
                            N: int,
                            stride_qn: int, stride_qd: int,
                            stride_kn: int, stride_kd: int,
                            stride_vn: int, stride_vd: int,
                            stride_on: int, stride_od: int,
                            d: tl.constexpr,
                            B_c: tl.constexpr, 
                            B_r: tl.constexpr,
                            ) -> torch.Tensor:
    pid = tl.program_id(axis=0)
    T_r = tl.cdiv(N, B_r)
    T_c = tl.cdiv(N, B_c)

    #      [1] + ([B_r, 1] + [1, d])

    # This code is as per Algorithm 1 in Flash Attention 2: https://arxiv.org/pdf/2307.08691
    ##  3: for 1 <= i <= T_r do
    off_r = (pid * B_r + tl.arange(0, B_r)) % N      # off_r has shape [B_r]
    off_c = tl.arange(0, B_c)                        # off_c has shape [B_c]
    off_d = tl.arange(0, d)                          # off_d has shape [d]

    q_ptrs = q_ptr + (off_r[:, None] * stride_qn + off_d[None, :] * stride_qd)
    o_ptrs = o_ptr + (off_r[:, None] * stride_on + off_d[None, :] * stride_od)

    ## 4: load Q_i from HBM to on-chip SRAM
    q_i = tl.load(q_ptrs, mask=off_r[:, None] < N, other=0.0)   # q_i   has shape [B_r, d]

    ## 5: on-chip, initialize o_prev_i, l_prev_i, m_prev_i
    o_prev = tl.zeros((B_r, d), dtype=tl.float32)             # 5: o(0)_i has 0s    of shape [B_r, d]
    l_prev = tl.zeros((B_r, 1), dtype=tl.float32)             # 5: l(0)_i has 0s    of shape [B_r]
    m_prev = tl.full((B_r, 1), -math.inf, dtype=tl.float32)   # 5: m(0)_i has -infs of shape [B_r]
    
    # online softmax
    ## 6: for 1 <= j <= T_c do
    k_ptrs = k_ptr + off_c[:, None] * stride_kn + off_d[None, :] * stride_kd
    v_ptrs = v_ptr + off_c[:, None] * stride_vn + off_d[None, :] * stride_vd
    
    for j in tl.range(0, T_c):
        k_j = tl.load(k_ptrs, mask=off_c[:, None] < N - j * B_c, other=0.0)  ## 7: load K_j from HBM to on-chip SRAM. K_j has shape [B_c, d]
        v_j = tl.load(v_ptrs, mask=off_c[:, None] < N - j * B_c, other=0.0)  ## 7: load V_j from HBM to on-chip SRAM. V_j has shape [B_c, d]

        # there is no matmul op in triton
        s_i = tl.dot(q_i, k_j.T)                                         ## 8:  On chip, compute S(j)_i = ${ Q_i @ K_j.T                      }$ of shape [B_r, B_c]
        m_i = tl.maximum(m_prev, tl.max(s_i, axis=-1, keep_dims=True))   ## 9:  On chip, compute m(j)_i = ${ max( m(j-1)_i, rowmax(S(j)_i) )  }$ of shape [B_r]
        p_i = tl.exp(s_i - m_i)                                          ## 9:  On chip, compute P(j)_i = ${ exp( S(j)_i - m(j)_i )           }$ of shape [B_r, B_c]
        
        alpha = tl.exp(m_prev - m_i)
        l_i = alpha * l_prev + tl.sum(p_i, axis=-1, keep_dims=True)  ## 9:  On chip, compute l(j)_i = ${ exp( m(j-1)_i - m(j)_i ) * l(j)_i + rowsum(P(j)_i) }$ of shape [B_r]
        o_i = tl.dot(p_i, v_j, alpha * o_prev)                      ## 10: On chip, compute O(j)_i = ${ diag(exp( m(j-1)_i - m(j)_i )-1 + P(j)_i @ V_j )     }$ of shape [B_r, d]

        m_prev = m_i
        l_prev = l_i
        o_prev = o_i

        k_ptrs += B_c * stride_kn
        v_ptrs += B_c * stride_vn
        
    res = o_prev / l_prev      ## 11: On chip, compute O_i = O(j)_i / l(j)_i
    tl.store(o_ptrs, res)       

    # q_ptrs += B_r * stride_qn
    # o_ptrs += B_r * stride_on   

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    N,d = q.shape
    assert N == k.shape[0] and d == k.shape[1]
    o = torch.zeros((N,d), device=DEVICE, dtype=q.dtype)

    grid = lambda META: (triton.cdiv(N, META['B_r']), )
    flash_attention_kernel[grid](
        q,k,v,o,
        N,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
    )
    return o

def main():
    seq_len = 1024
    head_dim = 128

    q = torch.randn((seq_len, head_dim), device=DEVICE, dtype=torch.float32)
    k = torch.randn((seq_len, head_dim), device=DEVICE, dtype=torch.float32)
    v = torch.randn((seq_len, head_dim), device=DEVICE, dtype=torch.float32)

    flash_output = flash_attention(q,k,v)
    torch_output = torch_attention(q,k,v)
    torch_flash_output = torch_flash_attention(q,k,v)
    rtol = 0
    if torch.allclose(flash_output, torch_output, atol=1e-2, rtol=rtol):
        print("✅ Flash and Base match")
    else:
        print("❌ Flash and Base differ")
        to_csv = lambda tensor, csv : pandas.DataFrame(tensor.cpu().numpy()).to_csv(csv)

        to_csv(flash_output, 'flash.csv')
        to_csv(torch_output, 'torch.csv')
        to_csv(torch_flash_output, 'torch_flash.csv')

        diff = (flash_output - torch_output) / torch_output
        threshold = 0.1
        mask = torch.abs(diff) < threshold
        zeros = torch.zeros_like(diff)
        zeros[~mask] = diff[~mask]
        to_csv(zeros, 'diff.csv')

        # torch.save(flash_output, 'flash.pt')
        # torch.save(base_output, 'base.pt')

main()