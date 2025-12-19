"""
Hybrid NF4/FP4/AF4 Operations with legacy format support.

This module provides custom ops that handle both legacy (bitsandbytes-style) and 
new (comfy_quant) key formats for 4-bit quantized models.

Legacy format:
- .absmax - per-block scales
- .quant_map - 16-value codebook
- .quant_state.bitsandbytes__nf4 - JSON with quant_type, blocksize, dtype, shape

ComfyUI format:
- .absmax - per-block scales
- .comfy_quant - JSON with format, group_size, quant_type, dtype, shape, quant_map
"""

import torch
import logging
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, LAYOUTS

# Try to import bitsandbytes for efficient dequantization
try:
    import bitsandbytes.functional as bnb_F
    _HAS_BNB = True
    logging.info("bitsandbytes available for 4-bit dequantization")
except ImportError:
    _HAS_BNB = False
    logging.info("bitsandbytes not available, using fallback dequantization")

# Try to import our NF4/FP4 layouts
try:
    from .quant_layouts.nf4_layout import NF4Layout, _check_nf4_available, _get_nf4_function
    _HAS_NF4_LAYOUT = True
except ImportError:
    _HAS_NF4_LAYOUT = False
    logging.warning("NF4 layout not available")

try:
    from .quant_layouts.fp4_layout import FP4Layout, _check_fp4_available, _get_fp4_function
    _HAS_FP4_LAYOUT = True
except ImportError:
    _HAS_FP4_LAYOUT = False
    logging.warning("FP4 layout not available")


class HybridNF4Ops(manual_cast):
    """
    Hybrid NF4/FP4/AF4 operations class that handles legacy and comfy_quant formats.
    
    Automatically converts:
    - Legacy bitsandbytes quant_state format -> proper layout_params
    - ComfyUI comfy_quant format -> proper layout_params
    - Creates QuantizedTensor with NF4Layout or FP4Layout
    """
    
    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.absmax = None
            self.quant_map = None
            self.block_size = 64  # Default block size
            self.quant_type = None  # 'nf4', 'fp4', or 'af4'
            self.original_shape = None
            self.original_dtype = None
            self.is_quantized = False
            
        def reset_parameters(self):
            return None
        
        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            """
            Custom state dict loading that handles both legacy and comfy_quant formats.
            
            Legacy format keys:
            - {prefix}weight (uint8 packed)
            - {prefix}absmax (float32 scales)
            - {prefix}quant_map (float32 codebook)
            - {prefix}quant_state.bitsandbytes__nf4 (JSON tensor)
            
            ComfyUI format keys:
            - {prefix}weight (uint8 packed)
            - {prefix}absmax (float32 scales)
            - {prefix}comfy_quant (JSON tensor with format, group_size, quant_type, dtype, shape, quant_map)
            """
            weight_key = prefix + 'weight'
            
            # Pop absmax (used by both formats)
            absmax_key = prefix + 'absmax'
            absmax = state_dict.pop(absmax_key, None)
            
            # Pop quant_map if present
            quant_map_key = prefix + 'quant_map'
            quant_map = state_dict.pop(quant_map_key, None)
            
            # Try to parse comfy_quant metadata first
            comfy_quant_tensor = state_dict.pop(prefix + 'comfy_quant', None)
            quant_state_tensor = None
            
            # Check for legacy bitsandbytes format
            for key in list(state_dict.keys()):
                if key.startswith(prefix + 'quant_state.bitsandbytes__'):
                    quant_state_tensor = state_dict.pop(key)
                    # Extract quant_type from key name
                    self.quant_type = key.split('bitsandbytes__')[-1]
                    break
            
            # Parse metadata
            if comfy_quant_tensor is not None:
                try:
                    from .comfy_quant_helpers import tensor_to_dict
                    layer_conf = tensor_to_dict(comfy_quant_tensor)
                    self.block_size = layer_conf.get('group_size', 64)
                    self.quant_type = layer_conf.get('quant_type', None)
                    self.original_dtype = layer_conf.get('dtype', 'float16')
                    self.original_shape = layer_conf.get('shape', None)
                    
                    # Get quant_map from comfy_quant if not separately stored
                    if quant_map is None and 'quant_map' in layer_conf:
                        quant_map = torch.tensor(layer_conf['quant_map'], dtype=torch.float32)
                    
                    # Infer quant_type from format if not explicit
                    if self.quant_type is None:
                        fmt = layer_conf.get('format', '')
                        if 'nf4' in fmt:
                            self.quant_type = 'nf4'
                        elif 'fp4' in fmt:
                            self.quant_type = 'fp4'
                        elif 'af4' in fmt:
                            self.quant_type = 'af4'
                    
                    logging.debug(f"Parsed comfy_quant: quant_type={self.quant_type}, block_size={self.block_size}")
                except Exception as e:
                    logging.debug(f"Failed to parse comfy_quant metadata: {e}")
            
            elif quant_state_tensor is not None:
                # Parse legacy bitsandbytes format
                try:
                    from .comfy_quant_helpers import tensor_to_dict
                    quant_state = tensor_to_dict(quant_state_tensor)
                    self.block_size = quant_state.get('blocksize', 64)
                    self.original_dtype = quant_state.get('dtype', 'float16')
                    self.original_shape = quant_state.get('shape', None)
                    logging.debug(f"Parsed legacy quant_state: quant_type={self.quant_type}, blocksize={self.block_size}")
                except Exception as e:
                    logging.debug(f"Failed to parse legacy quant_state: {e}")
            
            # Remove input_scale if present (not used for 4-bit)
            state_dict.pop(prefix + 'input_scale', None)
            
            # Load weight tensor
            weight_tensor = state_dict.pop(weight_key, None)
            
            if weight_tensor is not None:
                # Check if this is a 4-bit quantized weight
                if weight_tensor.dtype == torch.uint8 and absmax is not None:
                    self.is_quantized = True
                    self.absmax = absmax
                    self.quant_map = quant_map
                    
                    # Determine layout type
                    if self.quant_type in ('nf4', 'af4'):
                        layout_type = "NF4Layout"
                    elif self.quant_type == 'fp4':
                        layout_type = "FP4Layout"
                    else:
                        # Default to NF4 if unknown
                        layout_type = "NF4Layout"
                        self.quant_type = 'nf4'
                    
                    # Build layout_params for QuantizedTensor
                    # Convert original_dtype string to torch dtype
                    orig_dtype = getattr(torch, self.original_dtype, torch.float16) if isinstance(self.original_dtype, str) else torch.float16
                    
                    layout_params = {
                        'absmax': absmax,
                        'block_size': self.block_size,
                        'orig_dtype': orig_dtype,
                        'shape': torch.Size(self.original_shape) if self.original_shape else weight_tensor.shape,
                        'code': quant_map,
                        'quant_type': self.quant_type,
                    }
                    
                    if layout_type in LAYOUTS:
                        self.weight = torch.nn.Parameter(
                            QuantizedTensor(weight_tensor, layout_type, layout_params),
                            requires_grad=False
                        )
                        logging.debug(f"Loaded 4-bit layer {weight_key} with layout {layout_type}")
                    else:
                        # Fallback: store raw tensor for manual dequantization
                        self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                        logging.debug(f"Loaded 4-bit layer {weight_key} without layout")
                else:
                    # Non-4-bit weight - high-precision layer
                    self.is_quantized = False
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                    logging.debug(f"Loaded high-precision layer {weight_key} dtype={weight_tensor.dtype}")
            else:
                missing_keys.append(weight_key)
            
            # Handle bias
            bias_key = prefix + 'bias'
            bias_tensor = state_dict.pop(bias_key, None)
            if bias_tensor is not None:
                self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None
        
        def _dequantize_weight(self, weight, input_dtype):
            """Dequantize 4-bit weight to float.
            
            Priority:
            1. QuantizedTensor.dequantize() if wrapped in QuantizedTensor
            2. bitsandbytes.functional.dequantize_4bit if available
            3. Pure PyTorch fallback (slower but no dependencies)
            """
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()
            
            # 4-bit dequantization required
            if weight.dtype == torch.uint8 and self.absmax is not None:
                absmax = self.absmax.to(weight.device)
                block_size = self.block_size
                shape = torch.Size(self.original_shape) if self.original_shape else weight.shape
                quant_type = self.quant_type if self.quant_type else 'nf4'
                
                # Method 1: Try bitsandbytes (most efficient)
                if _HAS_BNB:
                    try:
                        # Create a QuantState for bitsandbytes
                        quant_state = bnb_F.QuantState(
                            absmax=absmax,
                            shape=shape,
                            dtype=input_dtype,
                            blocksize=block_size,
                            quant_type=quant_type,
                        )
                        return bnb_F.dequantize_4bit(weight, quant_state, quant_type=quant_type)
                    except Exception as e:
                        logging.debug(f"bitsandbytes dequantize_4bit failed: {e}, falling back to PyTorch")
                
                # Method 2: Pure PyTorch fallback
                # Get codebook
                if self.quant_map is not None:
                    code = self.quant_map.to(weight.device).to(input_dtype)
                else:
                    if quant_type in ('nf4', 'af4'):
                        code = torch.tensor([
                            -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                            -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                            0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                            0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
                        ], device=weight.device, dtype=input_dtype)
                    else:  # fp4
                        code = torch.tensor([
                            0, 0.0625, 8.0, 12.0, 4.0, 6.0, 2.0, 3.0,
                            -0, -0.0625, -8.0, -12.0, -4.0, -6.0, -2.0, -3.0
                        ], device=weight.device, dtype=input_dtype)
                        code = code / code.abs().max()  # Normalize
                
                # Unpack nibbles
                A = weight.reshape(-1)
                n_unpacked = A.size(0) * 2
                out_indices = torch.empty(n_unpacked, dtype=torch.int32, device=weight.device)
                out_indices[::2] = (A >> 4).to(torch.int32)
                out_indices[1::2] = (A & 0xF).to(torch.int32)
                
                # Codebook lookup
                out_dq = code[out_indices]
                
                # Trim to output size
                n = shape.numel()
                if out_dq.numel() > n:
                    out_dq = out_dq[:n]
                
                # Apply blockwise scales
                blocks = n // block_size
                rem = n % block_size
                has_rem = rem > 0
                
                out = torch.empty(n, dtype=input_dtype, device=weight.device)
                if blocks > 0:
                    out[:blocks * block_size] = (
                        out_dq[:blocks * block_size].view(-1, block_size) 
                        * absmax[:blocks].view(-1, 1)
                    ).reshape(-1)
                if has_rem:
                    out[-rem:] = out_dq[-rem:] * absmax[-1]
                
                return out.view(shape).to(input_dtype)
            
            return weight.to(input_dtype)
        
        def forward_comfy_cast_weights(self, input):
            """Forward pass with proper 4-bit handling."""
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data
            
            input_dtype = input.dtype
            
            # Handle QuantizedTensor (triggers dispatch)
            if isinstance(weight, QuantizedTensor):
                # Move to input device if needed
                if weight.device != input.device:
                    weight = weight.to(device=input.device)
                
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input_dtype)
                
                # This triggers QuantizedTensor dispatch -> layout handler
                return torch.nn.functional.linear(input, weight, bias)
            
            # Fallback: dequantize weight manually
            if self.is_quantized and weight.dtype == torch.uint8:
                weight = weight.to(device=input.device)
                weight_dequant = self._dequantize_weight(weight, input_dtype)
                
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input_dtype)
                return torch.nn.functional.linear(input, weight_dequant, bias)
            
            # Standard manual_cast path for non-quantized weights
            weight, bias, offload_stream = cast_bias_weight(self, input, offloadable=True)
            out = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out
        
        def forward(self, *args, **kwargs):
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                weight = self.weight
                if isinstance(weight, torch.nn.Parameter):
                    weight = weight.data
                
                # 4-bit needs our special forward path
                if weight.dtype == torch.uint8 or isinstance(weight, QuantizedTensor):
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                return super().forward(*args, **kwargs)
        
        def convert_weight(self, weight, inplace=False, **kwargs):
            """Convert weight for LoRA patching - dequantize 4-bit."""
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()
            
            if weight.dtype == torch.uint8:
                return self._dequantize_weight(weight, torch.float32)
            
            return weight
        
        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            """Set weight after LoRA patching."""
            if return_weight:
                return weight
            
            if inplace_update:
                self.weight.data.copy_(weight)
            else:
                self.weight = torch.nn.Parameter(weight, requires_grad=False)
            
            # Mark as no longer quantized after patching
            self.is_quantized = False
            self.absmax = None
            self.quant_map = None
    
    # Normalization layers - use standard manual_cast versions
    class GroupNorm(manual_cast.GroupNorm):
        pass
    
    class LayerNorm(manual_cast.LayerNorm):
        pass
    
    class RMSNorm(manual_cast.RMSNorm):
        pass
    
    # Convolution layers - use standard manual_cast versions
    class Conv1d(manual_cast.Conv1d):
        pass
    
    class Conv2d(manual_cast.Conv2d):
        pass
    
    class Conv3d(manual_cast.Conv3d):
        pass
    
    class ConvTranspose1d(manual_cast.ConvTranspose1d):
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
