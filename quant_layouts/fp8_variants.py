"""
FP8 Variant Layouts (Row-wise and Block-wise scaling)

These layouts extend ComfyUI's base TensorCoreFP8Layout with finer-grained scaling:
- RowWiseFP8Layout: One scale per row (M scales for MxN weight)
- BlockWiseFP8Layout: One scale per 2D block (MxN blocks)

Both use dequantize-fallback for inference since torch._scaled_mm doesn't support
per-row or per-block scales directly.
"""

import torch
import logging
from typing import Tuple, Dict

# Import from ComfyUI core
from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_op


class RowWiseFP8Layout(QuantizedLayout):
    """
    Row-wise FP8 quantization layout.

    Storage format:
    - qdata: FP8 tensor (torch.float8_e4m3fn)
    - scale: Per-row scaling factors, shape (out_features,) - stored as dequant scale
    - orig_dtype: Original dtype before quantization
    """

    @classmethod
    def quantize(
        cls, tensor, scale=None, dtype=torch.float8_e4m3fn, **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize a 2D tensor with row-wise scaling.
        """
        orig_dtype = tensor.dtype

        if tensor.ndim != 2:
            raise ValueError(
                f"RowWiseFP8Layout requires 2D tensor, got shape {tensor.shape}"
            )

        M, N = tensor.shape
        fp8_max = torch.finfo(dtype).max

        if scale is None:
            # Compute per-row absolute maximum
            row_max = tensor.abs().amax(dim=1, keepdim=True)  # (M, 1)
            quant_scale = fp8_max / row_max.clamp_min(1e-12)  # (M, 1)
        else:
            # scale is provided as dequant scale, convert to quant scale
            quant_scale = (
                (1.0 / scale).unsqueeze(1) if scale.ndim == 1 else (1.0 / scale)
            )

        # Apply scale per-row
        tensor_scaled = tensor * quant_scale

        # Clamp and convert
        tensor_scaled = torch.clamp(tensor_scaled, -fp8_max, fp8_max)
        qdata = tensor_scaled.to(dtype)

        # Store dequant scale (reciprocal of quant scale)
        dequant_scale = (1.0 / quant_scale).squeeze(1)  # (M,)

        layout_params = {
            "scale": dequant_scale.to(torch.float32),
            "orig_dtype": orig_dtype,
        }
        return qdata, layout_params

    @staticmethod
    def dequantize(qdata, scale, orig_dtype, **kwargs) -> torch.Tensor:
        """Dequantize FP8 tensor with row-wise scaling."""
        # Convert to target dtype (matching core ComfyUI pattern)
        plain_tensor = torch.ops.aten._to_copy.default(qdata, dtype=orig_dtype)
        # Cast scale to orig_dtype before in-place multiply to preserve output dtype
        scale_broadcast = scale.to(
            dtype=orig_dtype, device=plain_tensor.device
        ).unsqueeze(1)  # (M, 1)
        plain_tensor.mul_(scale_broadcast)
        return plain_tensor

    @classmethod
    def get_plain_tensors(cls, qtensor) -> torch.Tensor:
        """Extract raw tensors for computation."""
        return qtensor._qdata, qtensor._layout_params["scale"]


class BlockWiseFP8Layout(QuantizedLayout):
    """
    True 2D block-wise FP8 quantization layout.

    Storage format:
    - qdata: FP8 tensor (torch.float8_e4m3fn)
    - scale: Per-block scaling factors, shape (M//block_size, N//block_size)
    - block_size: Size of quantization blocks
    - orig_dtype: Original dtype before quantization
    """

    @classmethod
    def quantize(
        cls, tensor, scale=None, block_size=64, dtype=torch.float8_e4m3fn, **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize a 2D tensor with 2D block-wise scaling.
        """
        orig_dtype = tensor.dtype

        if tensor.ndim != 2:
            raise ValueError(
                f"BlockWiseFP8Layout requires 2D tensor, got shape {tensor.shape}"
            )

        M, N = tensor.shape

        if M % block_size != 0 or N % block_size != 0:
            raise ValueError(
                f"BlockWiseFP8Layout requires dimensions divisible by block_size={block_size}. "
                f"Got shape ({M}, {N})"
            )

        fp8_max = torch.finfo(dtype).max

        # Reshape to 2D blocks
        tensor_blocked = tensor.reshape(
            M // block_size, block_size, N // block_size, block_size
        )
        tensor_blocked = tensor_blocked.permute(0, 2, 1, 3)  # (M//bs, N//bs, bs, bs)

        if scale is None:
            # Compute per-block absolute maximum
            block_max = tensor_blocked.abs().amax(dim=(2, 3))  # (M//bs, N//bs)
            quant_scale = fp8_max / block_max.clamp_min(1e-12)
        else:
            quant_scale = 1.0 / scale

        # Apply scale per-block
        scale_broadcast = quant_scale.unsqueeze(-1).unsqueeze(-1)
        tensor_scaled = tensor_blocked * scale_broadcast

        # Clamp and convert
        tensor_scaled = torch.clamp(tensor_scaled, -fp8_max, fp8_max)
        qdata_blocked = tensor_scaled.to(dtype)

        # Reshape back
        qdata = qdata_blocked.permute(0, 2, 1, 3).reshape(M, N)
        dequant_scale = 1.0 / quant_scale

        layout_params = {
            "scale": dequant_scale.to(torch.float32),
            "block_size": block_size,
            "orig_dtype": orig_dtype,
        }
        return qdata, layout_params

    @staticmethod
    def dequantize(qdata, scale, block_size, orig_dtype, **kwargs) -> torch.Tensor:
        """Dequantize FP8 tensor with 2D block-wise scaling."""
        M, N = qdata.shape

        # Reshape to blocks
        qdata_blocked = qdata.reshape(
            M // block_size, block_size, N // block_size, block_size
        )
        qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)

        # Convert to target dtype (matching core ComfyUI pattern)
        dequantized = torch.ops.aten._to_copy.default(qdata_blocked, dtype=orig_dtype)

        # Cast scale to orig_dtype before in-place multiply to preserve output dtype
        scale_broadcast = (
            scale.to(dtype=orig_dtype, device=dequantized.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        dequantized.mul_(scale_broadcast)

        # Reshape back
        dequantized = dequantized.permute(0, 2, 1, 3).reshape(M, N)
        return dequantized

    @classmethod
    def get_plain_tensors(cls, qtensor) -> torch.Tensor:
        """Extract raw tensors for computation."""
        return (
            qtensor._qdata,
            qtensor._layout_params["scale"],
            qtensor._layout_params["block_size"],
        )


# ==============================================================================
# Operation Handlers (dequant-fallback for both layouts)
# ==============================================================================


@register_layout_op(torch.ops.aten.linear.default, "RowWiseFP8Layout")
def rowwise_fp8_linear(func, args, kwargs):
    """Row-wise FP8 linear operation (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, "RowWiseFP8Layout")
def rowwise_fp8_mm(func, args, kwargs):
    """Row-wise FP8 matrix multiplication (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return func(input_tensor, weight)


@register_layout_op(torch.ops.aten.addmm.default, "RowWiseFP8Layout")
def rowwise_fp8_addmm(func, args, kwargs):
    """Row-wise FP8 addmm operation (dequant-fallback)."""
    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    return func(bias, input_tensor, weight, **kwargs)


@register_layout_op(torch.ops.aten.view.default, "RowWiseFP8Layout")
@register_layout_op(torch.ops.aten.t.default, "RowWiseFP8Layout")
def rowwise_fp8_func(func, args, kwargs):
    """Handle view/transpose for row-wise FP8 tensors."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        plain_input, scale = RowWiseFP8Layout.get_plain_tensors(input_tensor)
        ar = list(args)
        ar[0] = plain_input
        return QuantizedTensor(
            func(*ar, **kwargs), "RowWiseFP8Layout", input_tensor._layout_params
        )
    return func(*args, **kwargs)


@register_layout_op(torch.ops.aten.linear.default, "BlockWiseFP8Layout")
def blockwise_fp8_linear(func, args, kwargs):
    """Block-wise FP8 linear operation (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, "BlockWiseFP8Layout")
def blockwise_fp8_mm(func, args, kwargs):
    """Block-wise FP8 matrix multiplication (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]

    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()

    return func(input_tensor, weight)


@register_layout_op(torch.ops.aten.addmm.default, "BlockWiseFP8Layout")
def blockwise_fp8_addmm(func, args, kwargs):
    """Block-wise FP8 addmm operation (dequant-fallback)."""
    bias = args[0]
    input_tensor = args[1]
    weight = args[2]

    if isinstance(bias, QuantizedTensor):
        bias = bias.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()

    return func(bias, input_tensor, weight, **kwargs)


@register_layout_op(torch.ops.aten.view.default, "BlockWiseFP8Layout")
@register_layout_op(torch.ops.aten.t.default, "BlockWiseFP8Layout")
def blockwise_fp8_func(func, args, kwargs):
    """Handle view/transpose for block-wise FP8 tensors."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        plain_input, scale, block_size = BlockWiseFP8Layout.get_plain_tensors(
            input_tensor
        )
        ar = list(args)
        ar[0] = plain_input
        return QuantizedTensor(
            func(*ar, **kwargs), "BlockWiseFP8Layout", input_tensor._layout_params
        )
    return func(*args, **kwargs)
