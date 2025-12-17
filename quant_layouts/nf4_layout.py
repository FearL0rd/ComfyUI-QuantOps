"""
NF4 (Normal Float 4-bit) Quantization Layout

4-bit quantization using a codebook derived from the normal distribution.
Reference: QLoRA paper (https://arxiv.org/abs/2305.14314)

NOTE: NF4 kernels are lazy-imported - if not available, a helpful error is raised.
"""

import torch
import logging
from typing import Tuple, Dict

from comfy.quant_ops import QuantizedLayout, QuantizedTensor, register_layout_op

# Lazy import state
_nf4_available = None
_nf4_functions = {}


def _check_nf4_available():
    """Check and cache NF4 kernel availability."""
    global _nf4_available, _nf4_functions
    
    if _nf4_available is not None:
        return _nf4_available
    
    try:
        from ..kernels.nf4_kernels import (
            quantize_nf4,
            dequantize_nf4,
            QuantState4bit,
            NF4_CODEBOOK,
        )
        _nf4_functions['quantize_nf4'] = quantize_nf4
        _nf4_functions['dequantize_nf4'] = dequantize_nf4
        _nf4_functions['QuantState4bit'] = QuantState4bit
        _nf4_functions['NF4_CODEBOOK'] = NF4_CODEBOOK
        _nf4_available = True
        logging.info("NF4 kernels loaded successfully")
    except ImportError as e:
        _nf4_available = False
        logging.info(f"NF4 kernels not available: {e}")
    
    return _nf4_available


def _get_nf4_function(name):
    """Get an NF4 function, raising helpful error if not available."""
    if not _check_nf4_available():
        raise RuntimeError(
            f"NF4 kernels not available. "
            f"Ensure the kernels/nf4_kernels.py file is present and has no import errors."
        )
    return _nf4_functions[name]


class NF4Layout(QuantizedLayout):
    """
    NF4 (Normal Float 4-bit) quantization layout.
    Uses a 16-value codebook derived from the normal distribution.
    """
    
    @classmethod
    def quantize(cls, tensor, block_size=64, compress_statistics=False, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Quantize tensor to NF4 format."""
        if not _check_nf4_available():
            raise RuntimeError("NF4 kernels not available")
        
        quantize_nf4 = _get_nf4_function('quantize_nf4')
        
        orig_dtype = tensor.dtype
        original_shape = tensor.shape
        packed, quant_state = quantize_nf4(tensor, block_size, compress_statistics)
        
        layout_params = {
            'absmax': quant_state.absmax,
            'block_size': block_size,
            'orig_dtype': orig_dtype,
            'shape': original_shape,
            'quant_type': 'nf4',
            'code': quant_state.code,
        }
        return packed, layout_params
    
    @staticmethod
    def dequantize(qdata, absmax, block_size, orig_dtype, shape, code=None, **kwargs) -> torch.Tensor:
        """Dequantize NF4 packed data back to float."""
        if not _check_nf4_available():
            raise RuntimeError("NF4 kernels not available")
        
        dequantize_nf4 = _get_nf4_function('dequantize_nf4')
        QuantState4bit = _get_nf4_function('QuantState4bit')
        NF4_CODEBOOK = _get_nf4_function('NF4_CODEBOOK')
        
        if code is None:
            code = NF4_CODEBOOK.to(qdata.device)
        
        quant_state = QuantState4bit(
            absmax=absmax, shape=shape, code=code,
            blocksize=block_size, quant_type='nf4', dtype=orig_dtype,
        )
        return dequantize_nf4(qdata, quant_state, orig_dtype)
    
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
    
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
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
