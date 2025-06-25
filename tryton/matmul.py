import torch

import triton 
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna2():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'B_M': 128, 'B_N': 256, 'B_K': 64, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'B_M': 64, 'B_N': 256, 'B_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'B_M': 128, 'B_N': 128, 'B_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'B_M': 128, 'B_N': 64, 'B_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'B_M': 64, 'B_N': 128, 'B_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'B_M': 128, 'B_N': 32, 'B_K': 32, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'B_M': 64, 'B_N': 32, 'B_K': 32, 'GROUP_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'B_M': 32, 'B_N': 64, 'B_K': 32, 'GROUP_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'B_M': 128, 'B_N': 256, 'B_K': 128, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'B_M': 256, 'B_N': 128, 'B_K': 128, 'GROUP_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'B_M': 256, 'B_N': 64, 'B_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'B_M': 64, 'B_N': 256, 'B_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'B_M': 128, 'B_N': 128, 'B_K': 128, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'B_M': 128, 'B_N': 64, 'B_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'B_M': 64, 'B_N': 128, 'B_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'B_M': 128, 'B_N': 32, 'B_K': 64, 'GROUP_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'B_M': 128, 'B_N': 256, 'B_K': 16, 'GROUP_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'B_M': 256, 'B_N': 256, 'B_K': 16, 'GROUP_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'B_M': 128, 'B_N': 128, 'B_K': 32, 'GROUP_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'B_M': 64, 'B_N': 128, 'B_K': 32, 'GROUP_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'B_M': 64, 'B_N': 64, 'B_K': 32, 'GROUP_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
# C = AB 
def matmul_kernel(
    a_ptr: torch.Tensor, # M x K
    b_ptr: torch.Tensor, # K x N
    c_ptr: torch.Tensor, # M x N
    M: int,
    N: int, 
    K: int,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    B_M: tl.constexpr,
    B_N: tl.constexpr,
    B_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    T_M = tl.cdiv(M, B_M)
    T_N = tl.cdiv(N, B_N)
    T_K = tl.cdiv(K, B_K)
    group_size = GROUP_M * T_N # what is this calculation? why group_m * T_n?
    gid = pid // group_size
    lid = pid % group_size     # local id in group

    start_m = gid * GROUP_M
    GROUP_M = min(group_size - start_m, GROUP_M)
    
    lid_m = start_m + (lid % GROUP_M)
    lid_n = lid // GROUP_M
    
    # create tensors
    offs_am = (lid_m * B_M + tl.arange(0, B_M)) % M # offs_am has shape [B_M]
    offs_bn = (lid_n * B_N + tl.arange(0, B_N)) % N # offs_bn has shape [B_N]
    offs_k  = tl.arange(0, B_K)                     # offs_k has shape  [B_K]

    # a_ptrs is a tensor of [B_M, B_K] ptrs
    # b_ptrs is a tensor of [B_K, B_N] ptrs

    # for a tensor arr with shape (a), arr[:, None] broadcasts to a shape (a, 1)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) # [B_M, B_K]
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn) # [B_K, B_N]

    acc = tl.zeros((B_M, B_N), dtype=tl.float32)
    for k in range(0, T_K):
        max_k = K - k * B_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < max_k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < max_k, other=0.0)
        acc = tl.dot(a, b, acc) # fma
        a_ptrs += B_K * stride_ak
        b_ptrs += B_K * stride_bk
    
    c = acc.to(tl.float16)

    offs_cm = lid_m * B_M + tl.arange(0, B_M)
    offs_cn = lid_n * B_N + tl.arange(0, B_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, 
           b: torch.Tensor) -> torch.Tensor:
    assert a.shape[-1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M,N), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(M, META['B_M']) * triton.cdiv(N, META['B_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

torch.manual_seed(0)
a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
rtol = 1e-2 if is_hip_cdna2() else 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")