"""
Tensorwise INT8 Operations

Custom ComfyUI operations for tensorwise INT8 quantization.
Uses torch._int_mm with dynamic per-row activation quantization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch import Tensor

from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight

# Import quantization utilities
from ..quant_layouts.tensorwise_int8_layout import (
    quantize_int8,
    quantize_int8_tensorwise,
    quantize_int8_axiswise,
    dequantize,
)


class TensorWiseInt8Ops(manual_cast):
    """
    Custom ComfyUI operations for tensorwise INT8 quantization.

    Uses torch._int_mm for native INT8 matmul with dynamic per-row
    activation quantization. Provides ~2x speedup on RTX 30-series.

    Usage:
        model_options = {"custom_operations": TensorWiseInt8Ops}
        model = comfy.sd.load_diffusion_model(path, model_options=model_options)
    """

    class Linear(manual_cast.Linear):
        """Linear layer with direct INT8 weight loading and fast _int_mm forward."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_scale = None
            self.input_scale = None  # Optional: for static activation quant
            self._is_quantized = False
            self.compute_dtype = torch.bfloat16

        def reset_parameters(self):
            """Skip weight initialization - we load from state dict."""
            return None

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            """
            Load INT8 weights and scales directly from state dict.
            No dequant/requant needed.
            """
            weight_key = prefix + "weight"
            scale_key = prefix + "weight_scale"
            input_scale_key = prefix + "input_scale"
            bias_key = prefix + "bias"

            # Pop scale tensors
            weight_scale = state_dict.pop(scale_key, None)
            input_scale = state_dict.pop(input_scale_key, None)

            # Pop comfy_quant metadata if present
            state_dict.pop(prefix + "comfy_quant", None)

            # Get weight tensor
            weight_tensor = state_dict.pop(weight_key, None)

            if weight_tensor is not None:
                if weight_tensor.dtype == torch.int8 and weight_scale is not None:
                    # Direct INT8 load
                    self._is_quantized = True
                    self.weight = nn.Parameter(weight_tensor, requires_grad=False)

                    # Store scale as scalar or tensor
                    if isinstance(weight_scale, torch.Tensor):
                        if weight_scale.numel() == 1:
                            self.weight_scale = weight_scale.float().item()
                        else:
                            self.weight_scale = weight_scale.float()
                    else:
                        self.weight_scale = float(weight_scale)

                    # Store input scale if present (for static quantization)
                    if input_scale is not None:
                        if isinstance(input_scale, torch.Tensor):
                            self.input_scale = input_scale.float()
                        else:
                            self.input_scale = torch.tensor(input_scale, dtype=torch.float32)

                elif weight_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    # High-precision weight - keep unquantized
                    self._is_quantized = False
                    self.weight = nn.Parameter(weight_tensor, requires_grad=False)
                else:
                    # Unknown dtype - store as-is
                    self._is_quantized = False
                    self.weight = nn.Parameter(weight_tensor, requires_grad=False)
            else:
                missing_keys.append(weight_key)

            # Handle bias
            bias_tensor = state_dict.pop(bias_key, None)
            if bias_tensor is not None:
                self.bias = nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None

        def forward(self, x: Tensor) -> Tensor:
            """Fast forward using torch._int_mm for quantized weights."""
            if not self._is_quantized:
                # Non-quantized path - use standard ComfyUI cast
                weight, bias, offload_stream = cast_bias_weight(
                    self, x, offloadable=True
                )
                out = F.linear(x, weight, bias)
                uncast_bias_weight(self, weight, bias, offload_stream)
                return out

            # Quantized path - use fast int8 matmul
            compute_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16

            # Flatten to 2D for matmul
            x_shape = x.shape
            x_2d = x.reshape(-1, x_shape[-1])

            # Small batch optimization (kernel overhead dominates)
            if x_2d.shape[0] <= 16:
                w_float = dequantize(self.weight, self.weight_scale).to(x.dtype)
                y = F.linear(x_2d, w_float, self.bias)
            else:
                # Dynamic or static activation quantization
                if self.input_scale is not None:
                    # Static quantization path
                    y = _int8_forward_static(
                        x_2d, self.weight, self.weight_scale,
                        self.input_scale, self.bias, compute_dtype
                    )
                else:
                    # Dynamic activation quantization (default)
                    y = _int8_forward_dynamic(
                        x_2d, self.weight, self.weight_scale,
                        self.bias, compute_dtype
                    )

            # Reshape back
            return y.reshape(*x_shape[:-1], y.shape[-1])

    # Non-Linear layers - use standard manual_cast versions
    class GroupNorm(manual_cast.GroupNorm):
        pass

    class LayerNorm(manual_cast.LayerNorm):
        pass

    class Conv2d(manual_cast.Conv2d):
        pass

    class Conv3d(manual_cast.Conv3d):
        pass

    class ConvTranspose2d(manual_cast.ConvTranspose2d):
        pass

    class Embedding(manual_cast.Embedding):
        pass

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        if dims == 2:
            return cls.Conv2d(*args, **kwargs)
        elif dims == 3:
            return cls.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


# ==============================================================================
# INT8 Forward Functions
# ==============================================================================


@torch.no_grad()
def _int8_forward_dynamic(
    x: Tensor,
    weight: Tensor,
    weight_scale: float,
    bias: Tensor | None,
    compute_dtype: torch.dtype
) -> Tensor:
    """
    Forward with dynamic per-row activation quantization.

    Args:
        x: Input tensor [M, K]
        weight: INT8 weight [N, K]
        weight_scale: Per-tensor weight scale (scalar)
        bias: Optional bias [N]
        compute_dtype: Output dtype

    Returns:
        Output tensor [M, N]
    """
    # Dynamic per-row activation quantization
    x_8, x_scale = quantize_int8_axiswise(x, dim=-1)

    # Try Triton kernel first
    try:
        from ..kernels.tensorwise_kernels import mm_8bit
        res = mm_8bit(x_8, weight.T)
    except ImportError:
        # Fallback to torch._int_mm
        res = torch._int_mm(x_8, weight.T)

    # Rescale and convert dtype
    res_scaled = res.float().mul_(weight_scale * x_scale).to(compute_dtype)

    # Add bias
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)

    return res_scaled


@torch.no_grad()
def _int8_forward_static(
    x: Tensor,
    weight: Tensor,
    weight_scale: float,
    input_scale: Tensor,
    bias: Tensor | None,
    compute_dtype: torch.dtype
) -> Tensor:
    """
    Forward with static (calibrated) activation quantization.

    Args:
        x: Input tensor [M, K]
        weight: INT8 weight [N, K]
        weight_scale: Per-tensor weight scale (scalar)
        input_scale: Calibrated input scale (scalar)
        bias: Optional bias [N]
        compute_dtype: Output dtype

    Returns:
        Output tensor [M, N]
    """
    # Static activation quantization using calibrated scale
    x_8 = quantize_int8(x, input_scale)

    # Try Triton kernel first
    try:
        from ..kernels.tensorwise_kernels import mm_8bit
        res = mm_8bit(x_8, weight.T)
    except ImportError:
        # Fallback to torch._int_mm
        res = torch._int_mm(x_8, weight.T)

    # Rescale: combined scale is weight_scale * input_scale
    res_scaled = res.float().mul_(weight_scale * input_scale).to(compute_dtype)

    # Add bias
    if bias is not None:
        res_scaled = res_scaled + bias.to(compute_dtype)

    return res_scaled
