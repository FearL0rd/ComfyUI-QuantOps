"""
Hybrid INT8 Operations with legacy format support.

This module provides custom ops that handle both legacy (scale_weight) and 
new (weight_scale) key formats for INT8 quantized models, similar to how 
ComfyUI_Hybrid-Scaled_fp8-Loader handles FP8 format conversion.
"""

import torch
import logging
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, LAYOUTS

# Try to import our INT8 layout
try:
    from .quant_layouts.int8_layout import BlockWiseINT8Layout, _check_triton_available
    _HAS_INT8_LAYOUT = True
except ImportError:
    _HAS_INT8_LAYOUT = False
    logging.warning("INT8 layout not available")


class HybridINT8Ops(manual_cast):
    """
    Hybrid INT8 operations class that handles legacy scale_weight format.
    
    Automatically converts:
    - scale_weight -> weight_scale on load
    - Creates QuantizedTensor with BlockWiseINT8Layout
    - Falls back to dequantize if no native INT8 matmul available
    """
    
    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scale_weight = None
            self.block_size = 128  # Default block size
            self.is_quantized = False
            
        def reset_parameters(self):
            return None
        
        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            """
            Custom state dict loading that handles both legacy and new scale key formats.
            
            Legacy format: {prefix}scale_weight
            New format: {prefix}weight_scale
            """
            weight_key = prefix + 'weight'
            
            # Get scale - handle both old (scale_weight) and new (weight_scale) key formats
            scale_weight_key_old = prefix + 'scale_weight'
            scale_weight_key_new = prefix + 'weight_scale'
            
            scale = state_dict.pop(scale_weight_key_old, None)
            if scale is None:
                scale = state_dict.pop(scale_weight_key_new, None)
            
            # Remove input_scale if present
            state_dict.pop(prefix + 'input_scale', None)
            state_dict.pop(prefix + 'scale_input', None)
            
            # Parse comfy_quant metadata for layout type and block_size
            comfy_quant_tensor = state_dict.pop(prefix + 'comfy_quant', None)
            self.quant_format = None  # Default: infer from scale shape
            
            if comfy_quant_tensor is not None:
                try:
                    import json
                    json_str = ''.join(chr(c) for c in comfy_quant_tensor.tolist())
                    metadata = json.loads(json_str)
                    self.block_size = metadata.get('params', {}).get('group_size', 128)
                    self.quant_format = metadata.get('format', None)  # 'int8_lodewise' or 'int8_blockwise'
                    logging.debug(f"Parsed comfy_quant: format={self.quant_format}, block_size={self.block_size}")
                except Exception as e:
                    logging.debug(f"Failed to parse comfy_quant metadata: {e}")
            
            # Load weight tensor
            weight_tensor = state_dict.pop(weight_key, None)
            
            if weight_tensor is not None:
                # Check if this is actually an INT8 quantized weight
                if weight_tensor.dtype == torch.int8:
                    self.is_quantized = True
                    self.scale_weight = scale
                    
                    if scale is not None and _HAS_INT8_LAYOUT:
                        # Create QuantizedTensor with BlockWiseINT8Layout
                        layout_params = {
                            'scale': scale.to(torch.float32),
                            'block_size': self.block_size,
                            'is_weight': True,
                            'orig_dtype': torch.bfloat16,  # Will be updated in forward
                        }
                        self.weight = torch.nn.Parameter(
                            QuantizedTensor(weight_tensor, "BlockWiseINT8Layout", layout_params),
                            requires_grad=False
                        )
                        logging.debug(f"Loaded INT8 layer {weight_key} with scale shape {scale.shape}")
                    else:
                        # Store raw INT8 tensor and scale for fallback dequantization
                        self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
                        logging.debug(f"Loaded INT8 layer {weight_key} without layout (scale={scale is not None})")
                else:
                    # Non-INT8 weight - this is a high-precision layer
                    self.is_quantized = False
                    self.scale_weight = None
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
        
        def _dequantize_weight(self, weight, scale, input_dtype):
            """Dequantize INT8 weight to float.
            
            Handles two INT8 formats:
            - Lodewise (int8_lodewise): scale shape (N, K//block_size) - per-row, per-K-block
            - Blockwise (int8_blockwise): scale shape (N//block_size, K//block_size) - 2D tile grid
            
            Uses quant_type from comfy_quant metadata if available, otherwise infers from scale shape.
            """
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()
            
            if weight.dtype == torch.int8 and scale is not None:
                N, K = weight.shape  # Weight is (out_features, in_features)
                block_size = self.block_size
                k_blocks = K // block_size if K % block_size == 0 else (K + block_size - 1) // block_size
                
                # Use quant_format from metadata if available
                quant_format = getattr(self, 'quant_format', None)
                
                # Lodewise dequantization
                if quant_format == 'int8_lodewise' or (quant_format is None and scale.ndim == 2 and scale.shape[0] == N):
                    if K % block_size == 0 and scale.shape == (N, k_blocks):
                        # Lodewise: scale shape (N, K//block_size) - per-row, per-K-block
                        qdata_blocked = weight.reshape(N, k_blocks, block_size)
                        scale_broadcast = scale.unsqueeze(-1).to(device=weight.device, dtype=input_dtype)
                        dequant = qdata_blocked.to(input_dtype) * scale_broadcast
                        return dequant.reshape(N, K)
                
                # Blockwise dequantization
                if quant_format == 'int8_blockwise' or (quant_format is None and scale.ndim == 2):
                    if N % block_size == 0 and K % block_size == 0:
                        expected_shape = (N // block_size, K // block_size)
                        if scale.shape == expected_shape:
                            qdata_blocked = weight.reshape(N // block_size, block_size, K // block_size, block_size)
                            qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)
                            scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1).to(device=weight.device, dtype=input_dtype)
                            dequant = qdata_blocked.to(input_dtype) * scale_broadcast
                            return dequant.permute(0, 2, 1, 3).reshape(N, K)
                
                # 1D scale fallback
                if scale.ndim == 1:
                    if scale.shape[0] == N:
                        # Per-row scaling
                        scale_broadcast = scale.unsqueeze(-1).to(device=weight.device, dtype=input_dtype)
                        return weight.to(input_dtype) * scale_broadcast
                    elif scale.shape[0] == 1:
                        # Per-tensor scaling
                        return weight.to(input_dtype) * scale.item()
                
                # Fallback: try broadcasting
                logging.warning(f"INT8 format={quant_format}, scale shape {scale.shape} for weight {weight.shape}, using broadcast")
                return weight.to(input_dtype) * scale.to(device=weight.device, dtype=input_dtype)
            
            return weight.to(input_dtype)
        
        def forward_comfy_cast_weights(self, input):
            """Forward pass with proper INT8 handling."""
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data
            
            input_dtype = input.dtype
            
            # Handle QuantizedTensor (new path - triggers dispatch)
            if isinstance(weight, QuantizedTensor):
                # Move to input device if needed
                if weight.device != input.device:
                    weight = weight.to(device=input.device)
                
                # Update orig_dtype
                if hasattr(weight, '_layout_params'):
                    weight._layout_params['orig_dtype'] = input_dtype
                
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input_dtype)
                
                # This triggers QuantizedTensor dispatch -> int8_linear handler
                return torch.nn.functional.linear(input, weight, bias)
            
            # Fallback: dequantize INT8 weight manually
            if self.is_quantized and weight.dtype == torch.int8:
                weight = weight.to(device=input.device)
                if self.scale_weight is not None:
                    scale = self.scale_weight.to(device=input.device)
                    weight_dequant = self._dequantize_weight(weight, scale, input_dtype)
                else:
                    weight_dequant = weight.to(input_dtype)
                
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
                
                # INT8 needs our special forward path
                if weight.dtype == torch.int8 or isinstance(weight, QuantizedTensor):
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                return super().forward(*args, **kwargs)
        
        def convert_weight(self, weight, inplace=False, **kwargs):
            """Convert weight for LoRA patching - dequantize INT8."""
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()
            
            if weight.dtype == torch.int8 and self.scale_weight is not None:
                return self._dequantize_weight(weight, self.scale_weight, torch.float32)
            
            return weight
        
        def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
            """Set weight after LoRA patching."""
            # For now, keep as dequantized (re-quantization is complex for INT8)
            if return_weight:
                return weight
            
            if inplace_update:
                self.weight.data.copy_(weight)
            else:
                self.weight = torch.nn.Parameter(weight, requires_grad=False)
            
            # Mark as no longer quantized after patching
            self.is_quantized = False
            self.scale_weight = None
    
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
