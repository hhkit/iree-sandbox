import math
import triton
import triton.language as tl
import torch

import iree.turbine.aot as aot
import iree.runtime as rt

# SRAM size
M = int(262144)

@triton.jit 
def attention(q: tl.tensor, k: tl.tensor, v: tl.tensor) -> tl.tensor:
    return tl.softmax(q @ k.T) @ v

@triton.jit
def flashattention(q: tl.tensor, k: tl.tensor, v: tl.tensor) -> tl.tensor:
    N = q.shape.index(0)
    d = float(q.shape.index(-1))
    B_c = int(math.ceil(float(M) / (4 * d)))
    B_r = int(min(B_c, d))
    T_r = int(math.ceil(float(N) / B_r))
    T_c = int(math.ceil(float(N) / B_c))

    o = tl.zeros(q.shape, dtype=tl.float16)
    l = tl.zeros((N), dtype=tl.float16)
    m = tl.full((N,), -math.inf, dtype=tl.float16)
    
    for j in tl.range(0, N, B_c):
        k_j = tl.load(k, [j, 0]).T
        v_j = tl.load(v, [j, 0])

        for i in tl.range(0, N, B_r):
            q_i = tl.load(q, [i, 0])
            o_i = tl.load(o, [i, 0])
            l_i = tl.load(l, [i, 0])
            m_i = tl.load(m, [i, 0])

            s = q_i @ k_j

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
