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
    """INT8 linear layer using torch.int8_mm for direct quantized matmul.
    
    Uses native torch.int8_mm which avoids materializing large float32 intermediates
    and handles scaling more efficiently than manual int32 -> float32 conversion.
    
    Ported from comfy-kitchen eager backend with OOM fixes.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    # Quantize input per-row
    x_8, x_scale = quantize_int8_rowwise(x_2d)

    # Compute using torch.int8_mm which is optimized for int8 operations
    # weight is [N, K], we need [K, N] for matmul so transpose
    result = torch.int8_mm(x_8, weight.T.contiguous())

    # Scale back efficiently: result * (weight_scale * x_scale)
    # Process in chunks to avoid materializing large float32 tensor
    # which causes OOM for large models
    
    M, N = result.shape
    chunk_size = max(1, min(M, 256 * 1024 * 1024 // (N * 4)))  # Estimate safe chunk size
    
    scaled_parts = []
    for i in range(0, M, chunk_size):
        end_i = min(i + chunk_size, M)
        chunk = result[i:end_i].float()
        
        # Apply scales: chunk * (weight_scale * x_scale[i:end_i])
        chunk_scales = (weight_scale * x_scale[i:end_i])
        chunk_scaled = chunk * chunk_scales
        
        # Convert to output dtype immediately to free memory
        chunk_scaled = chunk_scaled.to(out_dtype)
        scaled_parts.append(chunk_scaled)
    
    result = torch.cat(scaled_parts, dim=0)

    if bias is not None:
        result = result + bias.to(device=result.device, dtype=result.dtype)

    return result.reshape(*orig_shape[:-1], weight.shape[0])
