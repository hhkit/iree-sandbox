import math
import torch
import torch.nn as nn

import iree.turbine.aot as aot
import iree.runtime as rt

# SRAM size
M = int(262144)


class FlashAttention2(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        N = q.shape[0]
        d = q.shape[-1]
        B_c = int(math.ceil(float(M) / (4 * d)))
        B_r = int(min(B_c, d))

        o = torch.zeros(q.shape, dtype=torch.float16)
        l = torch.zeros((N), dtype=torch.float16)
        m = torch.fill(torch.empty((N), dtype=torch.float16), -math.inf)

        T_r = int(math.ceil(float(N) / B_r))
        T_c = int(math.ceil(float(N) / B_c))

        q_blocks = q.split(B_r, dim=0)
        k_blocks = k.split(B_c, dim=0)
        v_blocks = v.split(B_c, dim=0)

        o_blocks = list(o.split(B_r, dim=0))
        l_blocks = list(l.split(B_r, dim=0))

        # Source: FlashAttention2 (doi/2307.08691) Algorithm 1

        for i in range(T_r):
            q_i = q_blocks[i]
            o_i = o_blocks[i]

            # log sum...?
            o_prev = torch.zeros((B_r, d), dtype=torch.float16)
            l_prev = torch.zeros((B_r), dtype=torch.float16)
            m_prev = torch.full((B_r, 1), -math.inf, dtype=torch.float16)

            # online softmax
            for j in range(T_c):
                k_j = k_blocks[j]
                v_j = v_blocks[j]

                s_i = q_i @ k_j.T
                m_tilde, _ = s_i.max(dim=-1, keepdim=True)
                m_i = torch.maximum(m_prev, m_tilde)
                p_i = torch.exp(s_i - m_i)
                ex_m_diff = torch.exp(m_prev - m_i)
                l_i = ex_m_diff * l_prev + torch.sum(p_i, dim=-1, keepdim=True)
                # quantization of p_i would come in here
                #   easier to write in pytorch, but writing it this way makes the kernel
                # p_qi = p_i.to(torch.float16)
                o_i = ex_m_diff * o_prev + p_i @ v_j

                m_prev = m_i
                l_prev = l_i
                o_prev = o_i

            o_blocks[i] = o_i / l_prev
            l_blocks[i] = m_prev + torch.log(l_prev)

        return (torch.cat(o_blocks), torch.cat(l_blocks))


model = FlashAttention2()

seq_len = 1024
head_dim = 128

q = torch.zeros((seq_len, head_dim), dtype=torch.float16)
k = torch.zeros((seq_len, head_dim), dtype=torch.float16)
v = torch.zeros((seq_len, head_dim), dtype=torch.float16)

try:
    exported = aot.export(model, args=(q, k, v))
except ValueError:
    error_type, error_instance, traceback = sys.exc_info()
    print(error_instance.args[0])

exported.print_readable()
