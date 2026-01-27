"""
Tensorwise INT8 Triton Kernels

Autotuned 8-bit matmul kernel for fast INT8 inference.
Ported from ComfyUI-Flux2-INT8 (dxqb/OneTrainer).
"""

import torch
from torch import Tensor

# Lazy Triton import
_triton_available = None
_mm_kernel = None


def _check_triton_available():
    """Check and cache Triton availability."""
    global _triton_available, _mm_kernel

    if _triton_available is not None:
        return _triton_available

    try:
        import triton
        import triton.language as tl
        _triton_available = True
    except ImportError:
        _triton_available = False
        return False

    # Define the kernel inside the check function to avoid import errors
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        ],
        key=['QUANTIZED_M', 'N', 'K', 'stride_bk'],
    )
    @triton.jit
    def __mm_kernel(
            a_ptr, b_ptr, c_ptr,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
            QUANTIZED_M,
            FLOAT: tl.constexpr,
    ):
        """
        Autotuned 8-bit matmul kernel.

        Computes C = A @ B where A and B are int8 or float8.
        Output C is int32 (for int8) or float32 (for float8).
        """
        pid_n = tl.program_id(axis=0)
        pid_m = tl.program_id(axis=1)

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32 if FLOAT else tl.int32)

        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
            b_mask = (offs_bn[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_SIZE_K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32 if FLOAT else tl.int32)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    _mm_kernel = __mm_kernel
    return True


def mm_8bit_triton(a: Tensor, b: Tensor) -> Tensor:
    """
    Triton 8-bit matmul.

    Args:
        a: INT8 tensor [M, K]
        b: INT8 tensor [K, N]

    Returns:
        Result tensor [M, N] in int32 or float32
    """
    if not _check_triton_available():
        return None

    import triton

    FLOAT = (a.dtype == torch.float8_e4m3fn)
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: a.shape[1]={K} != b.shape[0]={K2}"

    c = torch.empty((M, N), device=a.device, dtype=torch.float32 if FLOAT else torch.int32)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(M, META['BLOCK_SIZE_M']))

    _mm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        QUANTIZED_M=M // 64, FLOAT=FLOAT
    )
    return c


def mm_8bit(a: Tensor, b: Tensor) -> Tensor:
    """
    8-bit matmul with automatic backend selection.

    Tries Triton first, falls back to torch._int_mm for int8.

    Args:
        a: INT8 or FP8 tensor [M, K]
        b: INT8 or FP8 tensor [K, N]

    Returns:
        Result tensor [M, N]
    """
    # Try Triton if available and on CUDA
    if _check_triton_available() and a.is_cuda:
        result = mm_8bit_triton(a, b)
        if result is not None:
            return result

    # Fallback to torch._int_mm for int8
    if a.dtype == torch.int8:
        return torch._int_mm(a, b)
    else:
        # FP8 fallback using scaled_mm with unit scales
        one = torch.ones(1, device=a.device)
        return torch._scaled_mm(a, b.T.contiguous().T, scale_a=one, scale_b=one)
