# Kernel utilities - lazy imports for optional Triton

# FP8 kernels (requires Triton)
try:
    from .fp8_kernels import (
        _check_triton_available as fp8_check_triton,
        fp8_act_quant,
        fp8_gemm_blockwise,
        fp8_addmm_blockwise,
        fp8_gemm_rowwise,
    )

    _HAS_FP8_KERNELS = fp8_check_triton()
except ImportError:
    _HAS_FP8_KERNELS = False

# Tensorwise INT8 kernels (Triton optional, falls back to torch._int_mm)
try:
    from .tensorwise_kernels import (
        mm_8bit,
        mm_8bit_triton,
        _check_triton_available as tensorwise_check_triton,
    )
    _HAS_TENSORWISE_KERNELS = True
except ImportError:
    _HAS_TENSORWISE_KERNELS = False

__all__ = [
    "_HAS_FP8_KERNELS",
    "_HAS_TENSORWISE_KERNELS",
    "fp8_act_quant",
    "fp8_gemm_blockwise",
    "fp8_addmm_blockwise",
    "fp8_gemm_rowwise",
    "mm_8bit",
]
