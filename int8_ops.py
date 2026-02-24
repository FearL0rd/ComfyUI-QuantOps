"""
Hybrid INT8 Operations with legacy format support.

This module provides custom ops that handle both legacy (scale_weight) and 
new (weight_scale) key formats for INT8 quantized models, similar to how 
ComfyUI_Hybrid-Scaled_fp8-Loader handles FP8 format conversion.
"""

import torch
import torch.nn.functional as F
import logging
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor
from comfy.model_patcher import LowVramPatch

# Try to import our INT8 layouts
try:
    from .quant_layouts.int8_layout import BlockWiseINT8Layout, _check_triton_available
    _HAS_INT8_LAYOUT = True
except ImportError:
    _HAS_INT8_LAYOUT = False
    logging.warning("INT8 blockwise layout not available")

try:
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
    _HAS_TENSORWISE_INT8_LAYOUT = True
except ImportError:
    _HAS_TENSORWISE_INT8_LAYOUT = False
    logging.warning("INT8 tensorwise layout not available from comfy_kitchen")


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
                    from .comfy_quant_helpers import tensor_to_dict
                    layer_conf = tensor_to_dict(comfy_quant_tensor)
                    # Flat structure: group_size at root level, not nested in params
                    self.block_size = layer_conf.get('group_size', 128)
                    self.quant_format = layer_conf.get('format', None)  # 'int8_lodewise' or 'int8_blockwise'
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
                    
                    if scale is not None:
                        is_tensorwise = self.quant_format == 'int8_tensorwise' or (self.quant_format is None and (scale.ndim == 0 or (scale.ndim == 1 and scale.shape[0] == 1)))
                        
                        if is_tensorwise and _HAS_TENSORWISE_INT8_LAYOUT:
                            layout_params = TensorWiseINT8Layout.Params(
                                scale=scale.to(torch.float32),
                                orig_dtype=torch.bfloat16,
                                orig_shape=tuple(weight_tensor.shape),
                                is_weight=True,
                            )
                            self.weight = torch.nn.Parameter(
                                QuantizedTensor(weight_tensor, "TensorWiseINT8Layout", layout_params),
                                requires_grad=False
                            )
                            logging.debug(f"Loaded INT8 layer {weight_key} with TensorWiseINT8Layout")
                        elif not is_tensorwise and _HAS_INT8_LAYOUT:
                            # Create QuantizedTensor with BlockWiseINT8Layout
                            layout_params = BlockWiseINT8Layout.Params(
                                scale=scale.to(torch.float32),
                                orig_dtype=torch.bfloat16,  # Will be updated in forward
                                orig_shape=tuple(weight_tensor.shape),
                                block_size=self.block_size,
                                is_weight=True,
                            )
                            self.weight = torch.nn.Parameter(
                                QuantizedTensor(weight_tensor, "BlockWiseINT8Layout", layout_params),
                                requires_grad=False
                            )
                            logging.debug(f"Loaded INT8 layer {weight_key} with BlockWiseINT8Layout")
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
                if hasattr(weight, '_params'):
                    object.__setattr__(weight._params, 'orig_dtype', input_dtype)
                
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
        
        def forward_fused_lora(self, input):
            """
            Memory-efficient LoRA forward pass.
            
            Computes: output = base_int8_matmul(x, W) + lora_contribution(x)
            
            Instead of dequantizing the full weight, we:
            1. Run native INT8 matmul for base (uses dynamic activation quant)
            2. Compute LoRA contribution separately: x @ B.T @ A.T * scale
            3. Sum the results
            
            This avoids materializing the full bf16 weight tensor.
            """
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data
            
            input_dtype = input.dtype
            
            # Log that we're using fused LoRA path (only first few times)
            if not hasattr(HybridINT8Ops.Linear, '_fused_lora_log_count'):
                HybridINT8Ops.Linear._fused_lora_log_count = 0
            if HybridINT8Ops.Linear._fused_lora_log_count < 3:
                logging.info(f"INT8: Using fused LoRA path - input={input.shape}, weight={weight.shape if hasattr(weight, 'shape') else weight._qdata.shape}")
                HybridINT8Ops.Linear._fused_lora_log_count += 1
            
            # 1. Base INT8 output (no LoRA applied, uses dynamic activation quant)
            if isinstance(weight, QuantizedTensor):
                # Move to input device if needed
                if weight.device != input.device:
                    weight = weight.to(device=input.device)
                
                # Update orig_dtype for proper output casting
                if hasattr(weight, '_params'):
                    object.__setattr__(weight._params, 'orig_dtype', input_dtype)
                
                # Dispatch to int8_linear handler (uses dynamic act quant now)
                base_out = torch.nn.functional.linear(input, weight, None)
            elif self.is_quantized and weight.dtype == torch.int8:
                # Fallback for raw INT8 without QuantizedTensor wrapper
                weight = weight.to(device=input.device)
                if self.scale_weight is not None:
                    scale = self.scale_weight.to(device=input.device)
                    weight_dequant = self._dequantize_weight(weight, scale, input_dtype)
                else:
                    weight_dequant = weight.to(input_dtype)
                base_out = F.linear(input, weight_dequant, None)
            else:
                # Non-quantized fallback
                base_out = F.linear(input.to(weight.dtype), weight, None)
            
            # 2. Compute LoRA contributions separately
            lora_out = None
            for patch_fn in self.weight_function:
                if isinstance(patch_fn, LowVramPatch):
                    # Extract patches for this layer
                    patches = patch_fn.patches.get(patch_fn.key, [])
                    for patch_data in patches:
                        # patch_data: (strength_patch, adapter, strength_model, offset, function)
                        strength_patch = patch_data[0]
                        adapter = patch_data[1]
                        strength_model = patch_data[2]
                        
                        # Check if adapter has weights (LoRA-style adapters)
                        if hasattr(adapter, 'weights') and adapter.weights is not None:
                            weights = adapter.weights
                            # weights[0] = mat1 (lora_up), weights[1] = mat2 (lora_down), weights[2] = alpha
                            mat1 = weights[0]  # [out_dim, rank]
                            mat2 = weights[1]  # [rank, in_dim]
                            alpha = weights[2] if weights[2] is not None else 1.0
                            rank = mat2.shape[0]
                            scale = strength_patch * strength_model * (alpha / rank)
                            
                            # Move to device
                            mat1 = mat1.to(device=input.device, dtype=input_dtype)
                            mat2 = mat2.to(device=input.device, dtype=input_dtype)
                            
                            # Compute: x @ mat2.T @ mat1.T * scale
                            # input: [B, seq, in_dim], mat2: [rank, in_dim], mat1: [out_dim, rank]
                            temp = F.linear(input, mat2)  # [B, seq, rank]
                            lora_contrib = F.linear(temp, mat1) * scale  # [B, seq, out_dim]
                            
                            if lora_out is None:
                                lora_out = lora_contrib
                            else:
                                lora_out = lora_out + lora_contrib
                        else:
                            # Fallback for non-LoRA adapters: apply the patch function to dequantized weight
                            # This is memory-heavy but ensures correctness
                            logging.warning(f"INT8 Fused LoRA: Falling back to dequant for non-LoRA adapter")
                            if isinstance(self.weight.data, QuantizedTensor):
                                weight_fp = self.weight.data.dequantize().to(input.device)
                            else:
                                weight_fp = self._dequantize_weight(
                                    self.weight.data.to(input.device),
                                    self.scale_weight.to(input.device) if self.scale_weight is not None else None,
                                    input_dtype
                                )
                            patched_weight = patch_fn(weight_fp)
                            # Compute the delta contribution
                            lora_contrib = F.linear(input, patched_weight - weight_fp, None)
                            if lora_out is None:
                                lora_out = lora_contrib
                            else:
                                lora_out = lora_out + lora_contrib
                else:
                    # Non-LowVramPatch function - fall back to calling it
                    logging.warning(f"INT8 Fused LoRA: Unknown patch function type, falling back")
                    if isinstance(self.weight.data, QuantizedTensor):
                        weight_fp = self.weight.data.dequantize().to(input.device)
                    else:
                        weight_fp = self._dequantize_weight(
                            self.weight.data.to(input.device),
                            self.scale_weight.to(input.device) if self.scale_weight is not None else None,
                            input_dtype
                        )
                    patched_weight = patch_fn(weight_fp)
                    lora_contrib = F.linear(input, patched_weight - weight_fp, None)
                    if lora_out is None:
                        lora_out = lora_contrib
                    else:
                        lora_out = lora_out + lora_contrib
            
            # 3. Combine base + LoRA
            out = base_out
            if lora_out is not None:
                out = out + lora_out
            
            # Add bias
            if self.bias is not None:
                bias = self.bias.to(device=input.device, dtype=input_dtype)
                out = out + bias
            
            return out
        
        def forward(self, *args, **kwargs):
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data
            
            # Check if we have LoRA patches AND quantized weight
            has_lora = len(self.weight_function) > 0
            is_quant = isinstance(weight, QuantizedTensor) or (self.is_quantized and weight.dtype == torch.int8)
            
            if has_lora and is_quant:
                # Use fused LoRA path to avoid full weight dequantization
                return self.forward_fused_lora(*args, **kwargs)
            elif self.comfy_cast_weights or has_lora or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
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
