"""
Tensorwise INT8 Quantization Layout

Per-tensor INT8 with scalar scale (W8A8 dynamic activation quantization).
Uses torch._int_mm for fast inference on all CUDA GPUs.
"""

import torch
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

# Import from ComfyUI core (which re-exports from comfy_kitchen)
from comfy.quant_ops import QuantizedTensor, register_layout_op

# Import base layout types from comfy_kitchen
try:
    from comfy_kitchen.tensor import QuantizedLayout, BaseLayoutParams
except ImportError:
    # Fallback for older versions
    from comfy.quant_ops import QuantizedLayout
    BaseLayoutParams = object


class TensorWiseInt8Layout(QuantizedLayout):
    """
    Per-tensor INT8 quantization layout with scalar scale.

    Storage format:
    - qdata: INT8 tensor (torch.int8)
    - scale: Single scalar scaling factor (float32)
    - orig_dtype: Original dtype before quantization
    - orig_shape: Original shape

    Uses dynamic per-row activation quantization at runtime.
    Inference via torch._int_mm for native INT8 matmul.
    """

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """Tensorwise INT8 layout parameters."""
        # scale and orig_dtype inherited from BaseLayoutParams if available
        pass

    @classmethod
    def quantize(
        cls, tensor: torch.Tensor, scale: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, "TensorWiseInt8Layout.Params"]:
        """
        Quantize a tensor to INT8 with per-tensor scaling.

        Args:
            tensor: Input tensor to quantize
            scale: Optional pre-computed scale (if None, computed from absmax)

        Returns:
            Tuple of (quantized_data, params)
        """
        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Compute per-tensor scale if not provided
        if scale is None:
            abs_max = tensor.abs().max()
            scale = (abs_max.float() / 127.0).clamp(min=1e-30)

        # Quantize: round and clamp to int8 range
        qdata = tensor.float().mul(1.0 / scale).round().clamp(-128.0, 127.0).to(torch.int8)

        params = cls.Params(
            scale=scale.to(torch.float32),
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )

        return qdata, params

    @staticmethod
    def dequantize(qdata: torch.Tensor, params) -> torch.Tensor:
        """Dequantize INT8 tensor back to original precision."""
        scale = params.scale
        orig_dtype = params.orig_dtype

        # Simple scalar multiplication
        return qdata.float() * scale.to(qdata.device)

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params) -> dict:
        """Return tensors for state dict serialization."""
        return {
            "": qdata,
            "weight_scale": params.scale,
        }

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor):
        """Extract raw tensors for computation."""
        return (
            qtensor._qdata,
            qtensor._params.scale,
        )


# ==============================================================================
# Quantization Utilities (ported from Flux2-INT8)
# ==============================================================================


try:
    from comfy_kitchen.backends.eager.quantization import (
        quantize_int8_tensorwise,
        quantize_int8_rowwise,
        dequantize_int8_simple as dequantize,
    )
except ImportError:
    from ..utils.eager_quantization import (
        quantize_int8_tensorwise,
        quantize_int8_rowwise,
        dequantize_int8_simple as dequantize,
    )

def quantize_int8(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantize tensor to int8 using provided scale."""
    # scale can be a float or a tensor
    s = scale.item() if isinstance(scale, torch.Tensor) and scale.numel() == 1 else scale
    return x.float().mul(1.0 / s).round().clamp(-128.0, 127.0).to(torch.int8)

def quantize_int8_axiswise(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to int8 with per-axis scale (for activations)."""
    if dim == -1 or dim == x.ndim - 1:
        return quantize_int8_rowwise(x)
    # Fallback for other dims
    abs_max = x.abs().amax(dim=dim, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    return quantize_int8(x, scale), scale


# ==============================================================================
# Operation Handlers
# ==============================================================================


@register_layout_op(torch.ops.aten.linear.default, TensorWiseInt8Layout)
def tensorwise_int8_linear(func, args, kwargs):
    """
    Tensorwise INT8 linear operation using torch.int8_mm.

    Uses dynamic per-row activation quantization with memory-efficient scaling.
    """
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    # Only weight is quantized (activations quantized dynamically)
    if not isinstance(weight, QuantizedTensor):
        # Fallback to standard linear
        return func(*args, **kwargs)

    weight_int8, weight_scale = TensorWiseInt8Layout.get_plain_tensors(weight)
    out_dtype = weight._params.orig_dtype

    # Ensure input is contiguous and on same device
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    if input_tensor.device != weight_int8.device:
        input_tensor = input_tensor.to(weight_int8.device)

    # Flatten to 2D for matmul
    orig_shape = input_tensor.shape
    x_2d = input_tensor.reshape(-1, orig_shape[-1])

    # Always use INT8 matmul (no dequantize fallback to prevent OOM)
    result = _tensorwise_int8_matmul(x_2d, weight_int8, weight_scale)

    # Cast to output dtype
    compute_dtype = input_tensor.dtype if input_tensor.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    result = result.to(compute_dtype)

    # Add bias
    if bias is not None:
        result = result + bias.to(result.device, dtype=result.dtype)

    # Reshape back
    return result.reshape(*orig_shape[:-1], result.shape[-1])


def _tensorwise_int8_matmul(x: torch.Tensor, weight_int8: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """
    Perform INT8 matmul with dynamic activation quantization.
    Uses torch.int8_mm with memory-efficient chunked scaling.

    Args:
        x: Input tensor [M, K] in float
        weight_int8: INT8 weight [N, K]
        weight_scale: Per-tensor weight scale (scalar)

    Returns:
        Output tensor [M, N] in float
    """
    # Try Triton kernel if available
    try:
        from ..kernels.tensorwise_kernels import mm_8bit_triton
        if mm_8bit_triton is not None and x.is_cuda:
            # Quantize activations per-row
            x_int8, x_scale = quantize_int8_axiswise(x, dim=-1)
            # Triton matmul
            result = mm_8bit_triton(x_int8, weight_int8.T)
            
            # Efficient scaling with chunking to avoid OOM
            M, N = result.shape
            chunk_size = max(1, min(M, 256 * 1024 * 1024 // (N * 4)))
            scaled_parts = []
            for i in range(0, M, chunk_size):
                end_i = min(i + chunk_size, M)
                chunk = result[i:end_i].float()
                chunk_scales = weight_scale * x_scale[i:end_i]
                chunk_scaled = chunk * chunk_scales
                scaled_parts.append(chunk_scaled)
            result = torch.cat(scaled_parts, dim=0)
            return result
    except ImportError:
        pass

    # Fallback to torch.int8_mm with efficient scaling
    x_int8, x_scale = quantize_int8_axiswise(x, dim=-1)
    result = torch.int8_mm(x_int8, weight_int8.T)
    
    # Efficient scaling with chunking to avoid materializing large float32
    M, N = result.shape
    chunk_size = max(1, min(M, 256 * 1024 * 1024 // (N * 4)))
    scaled_parts = []
    for i in range(0, M, chunk_size):
        end_i = min(i + chunk_size, M)
        chunk = result[i:end_i].float()
        chunk_scales = weight_scale * x_scale[i:end_i]
        chunk_scaled = chunk * chunk_scales
        scaled_parts.append(chunk_scaled)
    result = torch.cat(scaled_parts, dim=0)
    return result


@register_layout_op(torch.ops.aten.t.default, TensorWiseInt8Layout)
@register_layout_op(torch.ops.aten.view.default, TensorWiseInt8Layout)
def tensorwise_int8_view(func, args, kwargs):
    """Handle view/transpose for INT8 tensors."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        qdata = input_tensor._qdata
        ar = list(args)
        ar[0] = qdata
        new_qdata = func(*ar, **kwargs)
        return input_tensor._copy_with(qdata=new_qdata)
    return func(*args, **kwargs)


# Alias for compatibility with comfy-kitchen fork
TensorWiseINT8Layout = TensorWiseInt8Layout
