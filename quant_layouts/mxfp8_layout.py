"""
TensorCoreMXFP8Layout for ComfyUI-QuantOps.

Implements MXFP8 (Microscaling FP8) block quantization layout based on
the comfy-kitchen reference implementation:
- FP8 E4M3 data (torch.float8_e4m3fn)
- E8M0 block scales (power-of-2 exponents stored as uint8)
- Block size of 32 (fixed for MXFP8)
- Requires SM >= 10.0 (Blackwell) for hardware-accelerated matmul

For pre-Blackwell GPUs, this layout provides memory savings via
compressed storage, with dequantization at runtime.

This layout loads weights quantized by convert_to_quant --mxfp8.
Storage format:
  .weight       -> float8_e4m3fn (FP8 E4M3)
  .weight_scale -> uint8 (E8M0 block scales)
  .comfy_quant  -> JSON metadata tensor
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch

# Import from ComfyUI core
from comfy.quant_ops import QuantizedTensor, register_layout_op

# Import base layout types
try:
    from comfy_kitchen.tensor import QuantizedLayout, BaseLayoutParams
except ImportError:
    from comfy.quant_ops import QuantizedLayout
    BaseLayoutParams = object

# Check for comfy_kitchen MXFP8 support
try:
    import comfy_kitchen as ck
    HAS_CK_MXFP8 = hasattr(ck, 'dequantize_mxfp8')
except ImportError:
    HAS_CK_MXFP8 = False
    ck = None

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Constants
MXFP8_BLOCK_SIZE = 32
E8M0_BIAS = 127


def roundup(x: int, multiple: int) -> int:
    """Round up x to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


def e8m0_to_f32(e8m0_tensor: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 (uint8) exponent-only values to float32 scales."""
    exponents = e8m0_tensor.to(torch.int32) - E8M0_BIAS
    return (2.0 ** exponents.float())


class TensorCoreMXFP8Layout(QuantizedLayout):
    """MXFP8 block quantization with E8M0 (power-of-2) block scaling.

    Implements the same interface as comfy-kitchen's TensorCoreMXFP8Layout
    but with pure PyTorch dequantization for compatibility.

    Storage format:
    - qdata: FP8 E4M3 tensor (torch.float8_e4m3fn)
    - scale: E8M0 block scales as uint8, shape (M, N//32)
    - orig_dtype: Original dtype before quantization
    - orig_shape: Original tensor shape before padding

    Note:
        Requires SM >= 10.0 (Blackwell) for hardware-accelerated matmul.
        On older GPUs, dequantization fallback provides memory savings only.
    """

    MIN_SM_VERSION = (10, 0)

    @dataclass(frozen=True)
    class Params(BaseLayoutParams):
        """MXFP8 layout parameters.

        Inherits scale, orig_dtype, orig_shape from BaseLayoutParams.
        scale contains the E8M0 per-block scaling factors as uint8.
        """
        transposed: bool = False

    @classmethod
    def quantize(
        cls,
        tensor: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, "TensorCoreMXFP8Layout.Params"]:
        """Quantize a 2D tensor with MXFP8 block-wise E8M0 scaling.

        For runtime quantization; typically used for activations.
        Weights are pre-quantized by convert_to_quant.
        """
        if tensor.dim() != 2:
            raise ValueError(f"MXFP8 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)
        M, N = tensor.shape

        # Pad to 32x aligned
        padded_M = roundup(M, 32)
        padded_N = roundup(N, 32)
        if padded_M != M or padded_N != N:
            tensor = torch.nn.functional.pad(
                tensor, (0, padded_N - N, 0, padded_M - M)
            )
            M, N = tensor.shape

        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        num_blocks = N // MXFP8_BLOCK_SIZE

        # Compute per-block scales
        tensor_blocks = tensor.reshape(M, num_blocks, MXFP8_BLOCK_SIZE)
        block_max = tensor_blocks.abs().amax(dim=-1)  # (M, num_blocks)

        # E8M0: compute power-of-2 exponent
        scale_needed = block_max / fp8_max
        scale_needed = torch.clamp(scale_needed, min=2**(-127))
        log2_scale = torch.log2(scale_needed)
        exp_biased = torch.ceil(log2_scale).to(torch.int32) + E8M0_BIAS
        exp_biased = torch.clamp(exp_biased, 0, 254)
        block_scales_e8m0 = exp_biased.to(torch.uint8)

        # Reconstruct float scales for quantization
        block_scales_f32 = e8m0_to_f32(block_scales_e8m0)

        # Handle zero blocks
        zero_mask = (block_max == 0)
        block_scales_f32 = torch.where(
            zero_mask, torch.ones_like(block_scales_f32), block_scales_f32
        )

        # Quantize
        data_scaled = tensor_blocks.float() / block_scales_f32.unsqueeze(-1)
        data_scaled = torch.where(
            zero_mask.unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled
        )
        data_scaled = torch.clamp(data_scaled, -fp8_max, fp8_max)
        qdata = data_scaled.reshape(M, N).to(torch.float8_e4m3fn)

        params = cls.Params(
            scale=block_scales_e8m0,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )
        return qdata, params

    @classmethod
    def dequantize(cls, qdata: torch.Tensor, params: "TensorCoreMXFP8Layout.Params") -> torch.Tensor:
        """Dequantize MXFP8 tensor with E8M0 block-wise scaling.

        Uses comfy_kitchen.dequantize_mxfp8 when available, otherwise pure PyTorch.
        """
        scales_e8m0 = params.scale
        orig_dtype = params.orig_dtype
        orig_shape = params.orig_shape

        # Try to use comfy_kitchen dequantize if available
        if HAS_CK_MXFP8:
            try:
                # ck.dequantize_mxfp8 expects E8M0 scale (float8_e8m0fnu or viewed as such)
                # Our scales are stored as uint8, need to view as float8_e8m0fnu
                if scales_e8m0.dtype == torch.uint8:
                    scales_for_ck = scales_e8m0.view(torch.float8_e8m0fnu)
                else:
                    scales_for_ck = scales_e8m0
                
                result = ck.dequantize_mxfp8(qdata, scales_for_ck, orig_dtype)
                
                # Trim to original shape if padded
                orig_M, orig_N = orig_shape
                M, N = result.shape
                if M != orig_M or N != orig_N:
                    result = result[:orig_M, :orig_N]
                return result
            except Exception as e:
                logger.warning(f"ck.dequantize_mxfp8 failed, using PyTorch fallback: {e}")

        # Pure PyTorch fallback
        M, N = qdata.shape
        block_size = MXFP8_BLOCK_SIZE
        num_blocks = N // block_size

        # Convert E8M0 scales to float32
        # Handle case where scale tensor might be larger due to swizzle padding
        if scales_e8m0.shape[0] > M or scales_e8m0.shape[1] > num_blocks:
            scales_e8m0 = scales_e8m0[:M, :num_blocks]
        scales_f32 = e8m0_to_f32(scales_e8m0)

        # Dequantize
        qdata_f32 = qdata.float().reshape(M, num_blocks, block_size)
        dequantized = qdata_f32 * scales_f32.unsqueeze(-1)
        dequantized = dequantized.reshape(M, N)

        # Trim to original shape if padded
        orig_M, orig_N = orig_shape
        if M != orig_M or N != orig_N:
            dequantized = dequantized[:orig_M, :orig_N]

        return dequantized.to(orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor: QuantizedTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract raw tensors for computation."""
        return qtensor._qdata, qtensor._params.scale

    @classmethod
    def state_dict_tensors(cls, qdata: torch.Tensor, params: "TensorCoreMXFP8Layout.Params") -> dict:
        """Tensors to save in state dict."""
        return {"": qdata, "_scale": params.scale}

    @classmethod
    def get_padded_shape(cls, orig_shape: tuple) -> tuple:
        """Get padded shape for cuBLAS alignment."""
        if len(orig_shape) != 2:
            raise ValueError(f"MXFP8 requires 2D shape, got {len(orig_shape)}D")
        rows, cols = orig_shape
        return (roundup(rows, 32), roundup(cols, 32))

    @classmethod
    def supports_fast_matmul(cls) -> bool:
        """Check if hardware supports native MXFP8 matmul."""
        if not torch.cuda.is_available():
            return False
        sm_major, sm_minor = torch.cuda.get_device_capability()
        return (sm_major, sm_minor) >= cls.MIN_SM_VERSION


# ==============================================================================
# Operation Handlers - Dequantize fallback for all operations
# ==============================================================================

def _dequantize_args(args):
    """Dequantize any QuantizedTensor arguments."""
    result = []
    for arg in args:
        if isinstance(arg, QuantizedTensor):
            result.append(arg.dequantize())
        elif arg is None:
            result.append(None)
        else:
            result.append(arg)
    return result


@register_layout_op(torch.ops.aten.t.default, TensorCoreMXFP8Layout)
def _handle_mxfp8_transpose(func, args, kwargs):
    """Handle transpose as a logical flag flip for MXFP8."""
    input_tensor = args[0]
    if not isinstance(input_tensor, QuantizedTensor):
        return func(*args, **kwargs)

    old_shape = input_tensor._params.orig_shape
    new_params = TensorCoreMXFP8Layout.Params(
        orig_dtype=input_tensor._params.orig_dtype,
        orig_shape=(old_shape[1], old_shape[0]),
        scale=input_tensor._params.scale,
        transposed=not input_tensor._params.transposed,
    )
    return QuantizedTensor(input_tensor._qdata, "TensorCoreMXFP8Layout", new_params)


@register_layout_op(torch.ops.aten.mm.default, TensorCoreMXFP8Layout)
def _handle_mxfp8_mm(func, args, kwargs):
    """MXFP8 mm - dequantize fallback for pre-Blackwell GPUs."""
    # For now, always use dequantize fallback
    # Native support requires comfy_kitchen.scaled_mm_mxfp8 + Blackwell
    logger.debug("MXFP8 mm: Using dequant fallback")
    return func(*_dequantize_args(args))


@register_layout_op(torch.ops.aten.addmm.default, TensorCoreMXFP8Layout)
def _handle_mxfp8_addmm(func, args, kwargs):
    """MXFP8 addmm - dequantize fallback for pre-Blackwell GPUs."""
    logger.debug("MXFP8 addmm: Using dequant fallback")
    return func(*_dequantize_args(args), **kwargs)


@register_layout_op(torch.ops.aten.linear.default, TensorCoreMXFP8Layout)
def _handle_mxfp8_linear(func, args, kwargs):
    """MXFP8 linear: input @ weight.T + bias.

    Uses dequantize fallback on pre-Blackwell GPUs.
    Native MXFP8 matmul requires SM >= 10.0.
    """
    input_tensor, weight = args[0], args[1]
    bias = args[2] if len(args) > 2 else None

    # For now, always use dequantize fallback
    # Native support would require:
    # 1. comfy_kitchen.quantize_mxfp8 for input
    # 2. comfy_kitchen.scaled_mm_mxfp8 kernel
    # 3. Blackwell GPU (SM >= 10.0)
    logger.debug("MXFP8 linear: Using dequant fallback")
    return torch.nn.functional.linear(*_dequantize_args((input_tensor, weight, bias)))


@register_layout_op(torch.ops.aten.view.default, TensorCoreMXFP8Layout)
def _handle_mxfp8_view(func, args, kwargs):
    """Handle view for MXFP8 tensors - dequantize required."""
    input_tensor = args[0]
    if isinstance(input_tensor, QuantizedTensor):
        # View/reshape not supported for MXFP8 in quantized form
        return func(input_tensor.dequantize(), *args[1:], **kwargs)
    return func(*args, **kwargs)
