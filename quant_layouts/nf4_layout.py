"""
NF4 (Normal Float 4-bit) Quantization Layout

4-bit quantization using a codebook derived from the normal distribution.
Reference: QLoRA paper (https://arxiv.org/abs/2305.14314)

This layout uses bitsandbytes for dequantization when available,
with a pure PyTorch fallback for systems without bitsandbytes.
"""

import torch
import logging
from typing import Tuple, Dict

from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_op

# Try to import bitsandbytes for efficient dequantization
try:
    import bitsandbytes.functional as bnb_F
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False
    logging.info("bitsandbytes not available for NF4Layout, using pure PyTorch fallback")

# Standard NF4 codebook (from bitsandbytes)
NF4_CODEBOOK = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


def _dequantize_4bit_pytorch(qdata, absmax, block_size, shape, dtype, code=None):
    """Pure PyTorch 4-bit dequantization.
    
    Based on bitsandbytes _dequantize_4bit_impl:
    1. Unpack nibbles (two 4-bit values per byte)
    2. Use codebook lookup
    3. Multiply by absmax scales per block
    """
    # Get codebook
    if code is not None:
        code_tensor = code.to(qdata.device).to(dtype)
    else:
        code_tensor = torch.tensor(NF4_CODEBOOK, device=qdata.device, dtype=dtype)
    
    # Unpack nibbles
    A = qdata.reshape(-1)
    n_unpacked = A.size(0) * 2
    out_indices = torch.empty(n_unpacked, dtype=torch.int32, device=qdata.device)
    out_indices[::2] = (A >> 4).to(torch.int32)  # High nibble
    out_indices[1::2] = (A & 0xF).to(torch.int32)  # Low nibble
    
    # Codebook lookup
    out_dq = code_tensor[out_indices]
    
    # Trim to output size
    n = shape.numel() if isinstance(shape, torch.Size) else torch.Size(shape).numel()
    if out_dq.numel() > n:
        out_dq = out_dq[:n]
    
    # Apply blockwise scales
    absmax = absmax.to(qdata.device)
    blocks = n // block_size
    rem = n % block_size
    has_rem = rem > 0
    
    out = torch.empty(n, dtype=dtype, device=qdata.device)
    if blocks > 0:
        out[:blocks * block_size] = (
            out_dq[:blocks * block_size].view(-1, block_size) 
            * absmax[:blocks].view(-1, 1)
        ).reshape(-1)
    if has_rem:
        out[-rem:] = out_dq[-rem:] * absmax[-1]
    
    target_shape = shape if isinstance(shape, torch.Size) else torch.Size(shape)
    return out.view(target_shape)


class NF4Layout(QuantizedLayout):
    """
    NF4 (Normal Float 4-bit) quantization layout.
    Uses a 16-value codebook derived from the normal distribution.
    """
    
    @classmethod
    def quantize(cls, tensor, block_size=64, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Quantize tensor to NF4 format."""
        if not _HAS_BNB:
            raise RuntimeError("NF4 quantization requires bitsandbytes")
        
        orig_dtype = tensor.dtype
        original_shape = tensor.shape
        packed, quant_state = bnb_F.quantize_4bit(tensor, blocksize=block_size, quant_type='nf4')
        
        layout_params = {
            'absmax': quant_state.absmax,
            'block_size': block_size,
            'orig_dtype': orig_dtype,
            'shape': original_shape,
            'quant_type': 'nf4',
            'code': quant_state.code if hasattr(quant_state, 'code') else None,
        }
        return packed, layout_params
    
    @staticmethod
    def dequantize(qdata, absmax, block_size, orig_dtype, shape, code=None, quant_type='nf4', **kwargs) -> torch.Tensor:
        """Dequantize NF4 packed data back to float.
        
        Uses bitsandbytes when available, pure PyTorch otherwise.
        """
        # Ensure shape is a proper Size object
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        
        # Method 1: Try bitsandbytes (most efficient)
        if _HAS_BNB:
            try:
                quant_state = bnb_F.QuantState(
                    absmax=absmax.to(qdata.device),
                    shape=shape,
                    dtype=orig_dtype,
                    blocksize=block_size,
                    quant_type=quant_type,
                )
                return bnb_F.dequantize_4bit(qdata, quant_state, quant_type=quant_type)
            except Exception as e:
                logging.debug(f"bitsandbytes dequantize_4bit failed: {e}, falling back to PyTorch")
        
        # Method 2: Pure PyTorch fallback
        return _dequantize_4bit_pytorch(qdata, absmax, block_size, shape, orig_dtype, code)
    
    @classmethod
    def get_plain_tensors(cls, qtensor):
        """Extract raw tensors for computation."""
        return (
            qtensor._qdata,
            qtensor._layout_params['absmax'],
            qtensor._layout_params['block_size'],
            qtensor._layout_params['shape'],
        )


# ==============================================================================
# NF4 Operation Handlers (dequant-fallback, no native 4-bit matmul)
# ==============================================================================

@register_layout_op(torch.ops.aten.linear.default, "NF4Layout")
def nf4_linear(func, args, kwargs):
    """NF4 linear operation (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    
    # Dequantize weight
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    
    # Dequantize input if needed
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    
    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, "NF4Layout")
def nf4_mm(func, args, kwargs):
    """NF4 matrix multiplication (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]
    
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    
    return func(input_tensor, weight)


@register_layout_op(torch.ops.aten.addmm.default, "NF4Layout")
def nf4_addmm(func, args, kwargs):
    """NF4 addmm operation (dequant-fallback)."""
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
