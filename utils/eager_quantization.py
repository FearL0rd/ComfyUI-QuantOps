import torch
from typing import Tuple, Optional

def quantize_int8_tensorwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with single tensorwise scale.
    
    Ported from comfy-kitchen eager backend.
    """
    abs_max = x.abs().max()
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = (x.float() / scale).round().clamp(-128.0, 127.0).to(torch.int8)
    return q, scale

def quantize_int8_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to INT8 with per-row scales.
    
    Ported from comfy-kitchen eager backend.
    """
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    q = (x.float() / scale).round().clamp(-128.0, 127.0).to(torch.int8)
    return q, scale

def dequantize_int8_simple(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor with scale.
    
    Ported from comfy-kitchen eager backend.
    """
    return q.float() * scale

def int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """INT8 linear layer using torch._int_mm.
    
    Ported from comfy-kitchen eager backend.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    # Quantize input per-row
    x_8, x_scale = quantize_int8_rowwise(x_2d)

    # Compute: x_8 @ weight.T using torch._int_mm
    # weight is [N, K], we need [K, N] for matmul so transpose
    result = torch._int_mm(x_8, weight.T.contiguous())

    # Scale back: result * (weight_scale * x_scale)
    result = result.float() * (weight_scale * x_scale)

    if bias is not None:
        result = result + bias.to(result.dtype)

    result = result.to(out_dtype)
    return result.reshape(*orig_shape[:-1], weight.shape[0])
