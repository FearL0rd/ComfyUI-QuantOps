"""
FP4 (Floating Point 4-bit) Quantization Layout

4-bit quantization using a hardware-inspired floating point codebook.

NOTE: FP4 kernels are lazy-imported - if not available, a helpful error is raised.
"""

import torch
import logging
from typing import Tuple, Dict

from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_op

# Lazy import state
_fp4_available = None
_fp4_functions = {}


def _check_fp4_available():
    """Check and cache FP4 kernel availability."""
    global _fp4_available, _fp4_functions
    
    if _fp4_available is not None:
        return _fp4_available
    
    try:
        from ..kernels.nf4_kernels import (
            quantize_fp4,
            dequantize_fp4,
            QuantState4bit,
            FP4_CODEBOOK_NORMALIZED,
        )
        _fp4_functions['quantize_fp4'] = quantize_fp4
        _fp4_functions['dequantize_fp4'] = dequantize_fp4
        _fp4_functions['QuantState4bit'] = QuantState4bit
        _fp4_functions['FP4_CODEBOOK_NORMALIZED'] = FP4_CODEBOOK_NORMALIZED
        _fp4_available = True
        logging.info("FP4 kernels loaded successfully")
    except ImportError as e:
        _fp4_available = False
        logging.info(f"FP4 kernels not available: {e}")
    
    return _fp4_available


def _get_fp4_function(name):
    """Get an FP4 function, raising helpful error if not available."""
    if not _check_fp4_available():
        raise RuntimeError(
            f"FP4 kernels not available. "
            f"Ensure the kernels/nf4_kernels.py file is present and has no import errors."
        )
    return _fp4_functions[name]


class FP4Layout(QuantizedLayout):
    """
    FP4 (Floating Point 4-bit) quantization layout.
    Uses a 16-value codebook for hardware-inspired 4-bit float.
    """
    
    @classmethod
    def quantize(cls, tensor, block_size=64, compress_statistics=False, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Quantize tensor to FP4 format."""
        if not _check_fp4_available():
            raise RuntimeError("FP4 kernels not available")
        
        quantize_fp4 = _get_fp4_function('quantize_fp4')
        
        orig_dtype = tensor.dtype
        original_shape = tensor.shape
        packed, quant_state = quantize_fp4(tensor, block_size, compress_statistics)
        
        layout_params = {
            'absmax': quant_state.absmax,
            'block_size': block_size,
            'orig_dtype': orig_dtype,
            'shape': original_shape,
            'quant_type': 'fp4',
            'code': quant_state.code,
        }
        return packed, layout_params
    
    @staticmethod
    def dequantize(qdata, absmax, block_size, orig_dtype, shape, code=None, **kwargs) -> torch.Tensor:
        """Dequantize FP4 packed data back to float."""
        if not _check_fp4_available():
            raise RuntimeError("FP4 kernels not available")
        
        dequantize_fp4 = _get_fp4_function('dequantize_fp4')
        QuantState4bit = _get_fp4_function('QuantState4bit')
        FP4_CODEBOOK_NORMALIZED = _get_fp4_function('FP4_CODEBOOK_NORMALIZED')
        
        if code is None:
            code = FP4_CODEBOOK_NORMALIZED.to(qdata.device)
        
        quant_state = QuantState4bit(
            absmax=absmax, shape=shape, code=code,
            blocksize=block_size, quant_type='fp4', dtype=orig_dtype,
        )
        return dequantize_fp4(qdata, quant_state, orig_dtype)
    
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
# FP4 Operation Handlers (dequant-fallback)
# ==============================================================================

@register_layout_op(torch.ops.aten.linear.default, "FP4Layout")
def fp4_linear(func, args, kwargs):
    """FP4 linear operation (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else None
    
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    
    return torch.nn.functional.linear(input_tensor, weight, bias)


@register_layout_op(torch.ops.aten.mm.default, "FP4Layout")
def fp4_mm(func, args, kwargs):
    """FP4 matrix multiplication (dequant-fallback)."""
    input_tensor = args[0]
    weight = args[1]
    
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if isinstance(input_tensor, QuantizedTensor):
        input_tensor = input_tensor.dequantize()
    
    return func(input_tensor, weight)


@register_layout_op(torch.ops.aten.addmm.default, "FP4Layout")
def fp4_addmm(func, args, kwargs):
    """FP4 addmm operation (dequant-fallback)."""
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
