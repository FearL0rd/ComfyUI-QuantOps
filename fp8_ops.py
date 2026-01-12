"""
Hybrid FP8 Operations with correct block_size handling.

This module provides custom ops that correctly read group_size from per-layer
metadata for FP8 rowwise and blockwise quantized models.

The issue: Core ComfyUI's MixedPrecisionOps reads block_size from QUANT_ALGOS
fallback instead of per-layer metadata, causing wrong block boundaries.
"""

import json
import torch
import logging
from comfy.ops import manual_cast, cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, QUANT_ALGOS, get_layout_class


class HybridFP8Ops(manual_cast):
    """
    Hybrid FP8 operations class that correctly handles block_size from metadata.

    Fixes the core bug where block_size is read from QUANT_ALGOS fallback
    instead of per-layer .comfy_quant metadata.
    """

    class Linear(manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scale_weight = None
            self.block_size = None
            self.is_quantized = False
            self.layout_type = None
            self.quant_format = None

        def reset_parameters(self):
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
            Custom state dict loading that correctly reads group_size from per-layer metadata.
            """
            weight_key = prefix + "weight"

            # Get weight_scale
            scale = state_dict.pop(prefix + "weight_scale", None)

            # Remove input_scale if present (not used for weight dequantization)
            state_dict.pop(prefix + "input_scale", None)

            # Parse comfy_quant metadata for layout type and block_size
            comfy_quant_tensor = state_dict.pop(prefix + "comfy_quant", None)
            layer_conf = None

            if comfy_quant_tensor is not None:
                try:
                    # Decode the comfy_quant tensor to dict
                    layer_conf = json.loads(comfy_quant_tensor.numpy().tobytes())
                    self.quant_format = layer_conf.get("format", None)
                    # KEY FIX: Read group_size from per-layer metadata!
                    self.block_size = layer_conf.get("group_size", None)
                    logging.debug(
                        f"HybridFP8Ops: Parsed comfy_quant for {prefix}: format={self.quant_format}, group_size={self.block_size}"
                    )
                except Exception as e:
                    logging.debug(
                        f"HybridFP8Ops: Failed to parse comfy_quant metadata: {e}"
                    )

            # Load weight tensor
            weight_tensor = state_dict.pop(weight_key, None)

            if weight_tensor is not None:
                # Check if this is an FP8 tensor
                if weight_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    self.is_quantized = True
                    self.scale_weight = scale

                    # Determine layout type from format
                    if self.quant_format is not None:
                        qconfig = QUANT_ALGOS.get(self.quant_format, {})
                        self.layout_type = qconfig.get(
                            "comfy_tensor_layout", "TensorCoreFP8Layout"
                        )

                        # Fallback block_size from QUANT_ALGOS only if not in metadata
                        if self.block_size is None:
                            self.block_size = qconfig.get("group_size", None)
                    else:
                        # Infer layout from scale shape
                        if scale is not None:
                            if scale.ndim == 0 or (
                                scale.ndim == 1 and scale.numel() == 1
                            ):
                                self.layout_type = "TensorCoreFP8Layout"
                            elif (
                                scale.ndim == 1
                                and scale.numel() == weight_tensor.shape[0]
                            ):
                                self.layout_type = "RowWiseFP8Layout"
                            elif scale.ndim == 2:
                                self.layout_type = "BlockWiseFP8Layout"
                                # Infer block_size from scale shape
                                if self.block_size is None:
                                    M, N = weight_tensor.shape
                                    scale_M, scale_N = scale.shape
                                    if M % scale_M == 0 and N % scale_N == 0:
                                        self.block_size = M // scale_M
                            else:
                                self.layout_type = "TensorCoreFP8Layout"
                        else:
                            self.layout_type = "TensorCoreFP8Layout"

                    # Check if the layout is registered
                    try:
                        get_layout_class(self.layout_type)
                    except KeyError:
                        logging.warning(
                            f"HybridFP8Ops: Layout '{self.layout_type}' not registered, using TensorCoreFP8Layout"
                        )
                        self.layout_type = "TensorCoreFP8Layout"



                    # Create layout_params based on layout_type
                    if self.layout_type == "TensorCoreMXFP8Layout":
                        from .quant_layouts.mxfp8_layout import TensorCoreMXFP8Layout
                        # Get orig_dtype from comfy_quant metadata if available
                        orig_dtype_str = layer_conf.get("orig_dtype", "torch.bfloat16") if layer_conf else "torch.bfloat16"
                        if orig_dtype_str == "torch.bfloat16":
                            orig_dtype = torch.bfloat16
                        elif orig_dtype_str == "torch.float16":
                            orig_dtype = torch.float16
                        else:
                            orig_dtype = torch.bfloat16
                        
                        # Get orig_shape from metadata or use current shape
                        orig_shape = tuple(layer_conf.get("orig_shape", list(weight_tensor.shape))) if layer_conf else tuple(weight_tensor.shape)
                        
                        layout_params = TensorCoreMXFP8Layout.Params(
                            scale=scale,  # E8M0 as uint8
                            orig_dtype=orig_dtype,
                            orig_shape=orig_shape,
                        )
                        logging.debug(
                            f"HybridFP8Ops: Loading MXFP8 layer {prefix}, scale shape={scale.shape if scale is not None else None}"
                        )
                    elif self.layout_type == "BlockWiseFP8Layout":
                        from .quant_layouts.fp8_variants import BlockWiseFP8Layout
                        block_size = self.block_size if self.block_size is not None else 64
                        if self.block_size is None:
                            logging.warning(
                                f"HybridFP8Ops: No block_size found for {prefix}, using fallback 64"
                            )
                        layout_params = BlockWiseFP8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16,
                            orig_shape=tuple(weight_tensor.shape),
                            block_size=block_size,
                        )
                    elif self.layout_type == "RowWiseFP8Layout":
                        from .quant_layouts.fp8_variants import RowWiseFP8Layout
                        layout_params = RowWiseFP8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16,
                            orig_shape=tuple(weight_tensor.shape),
                        )
                    else:
                        # TensorCoreFP8Layout or other - use comfy's layout
                        from comfy.quant_ops import TensorCoreFP8Layout
                        layout_params = TensorCoreFP8Layout.Params(
                            scale=scale.to(torch.float32) if scale is not None else None,
                            orig_dtype=torch.bfloat16,
                            orig_shape=tuple(weight_tensor.shape),
                        )

                    # Create QuantizedTensor
                    self.weight = torch.nn.Parameter(
                        QuantizedTensor(weight_tensor, self.layout_type, layout_params),
                        requires_grad=False,
                    )
                    logging.debug(
                        f"HybridFP8Ops: Loaded FP8 layer {prefix} with layout={self.layout_type}, block_size={self.block_size}"
                    )
                else:
                    # Non-FP8 weight - high-precision layer
                    self.is_quantized = False
                    self.scale_weight = None
                    self.weight = torch.nn.Parameter(weight_tensor, requires_grad=False)
            else:
                missing_keys.append(weight_key)

            # Handle bias
            bias_key = prefix + "bias"
            bias_tensor = state_dict.pop(bias_key, None)
            if bias_tensor is not None:
                self.bias = torch.nn.Parameter(bias_tensor, requires_grad=False)
            else:
                self.bias = None

        def forward_comfy_cast_weights(self, input):
            """Forward pass with proper FP8 handling."""
            weight = self.weight
            if isinstance(weight, torch.nn.Parameter):
                weight = weight.data

            input_dtype = input.dtype

            # Handle QuantizedTensor (triggers dispatch to layout handlers)
            if isinstance(weight, QuantizedTensor):
                # Move to input device if needed
                if weight.device != input.device:
                    weight = weight.to(device=input.device)

                # Update orig_dtype for dequantization
                if hasattr(weight, "_params"):
                    object.__setattr__(weight._params, "orig_dtype", input_dtype)

                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input.device, dtype=input_dtype)

                # This triggers QuantizedTensor dispatch -> layout-specific handler
                return torch.nn.functional.linear(input, weight, bias)

            # Fallback: dequantize FP8 weight manually if needed
            if self.is_quantized and weight.dtype in [
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ]:
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
            weight, bias, offload_stream = cast_bias_weight(
                self, input, offloadable=True
            )
            out = torch.nn.functional.linear(input, weight, bias)
            uncast_bias_weight(self, weight, bias, offload_stream)
            return out

        def _dequantize_weight(self, weight, scale, input_dtype):
            """Dequantize FP8 weight to float.

            Handles:
            - TensorCoreFP8Layout: scalar scale
            - RowWiseFP8Layout: scale shape (M,)
            - BlockWiseFP8Layout: scale shape (M//block_size, N//block_size)
            """
            M, N = weight.shape

            # Scalar scale (tensor-wise)
            if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
                return weight.to(input_dtype) * scale.item()

            # Row-wise scale
            if scale.ndim == 1 and scale.shape[0] == M:
                scale_broadcast = scale.unsqueeze(1).to(
                    device=weight.device, dtype=input_dtype
                )
                return weight.to(input_dtype) * scale_broadcast

            # Block-wise scale
            if scale.ndim == 2 and self.block_size is not None:
                block_size = self.block_size
                if M % block_size == 0 and N % block_size == 0:
                    qdata_blocked = weight.reshape(
                        M // block_size, block_size, N // block_size, block_size
                    )
                    qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)
                    scale_broadcast = (
                        scale.unsqueeze(-1)
                        .unsqueeze(-1)
                        .to(device=weight.device, dtype=input_dtype)
                    )
                    dequant = qdata_blocked.to(input_dtype) * scale_broadcast
                    return dequant.permute(0, 2, 1, 3).reshape(M, N)

            # Fallback: try broadcasting
            logging.warning(
                f"FP8 scale shape {scale.shape} for weight {weight.shape}, using broadcast"
            )
            return weight.to(input_dtype) * scale.to(
                device=weight.device, dtype=input_dtype
            )

        def forward(self, *args, **kwargs):
            if (
                self.comfy_cast_weights
                or len(self.weight_function) > 0
                or len(self.bias_function) > 0
            ):
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                weight = self.weight
                if isinstance(weight, torch.nn.Parameter):
                    weight = weight.data

                # FP8 needs our special forward path
                if weight.dtype in [
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ] or isinstance(weight, QuantizedTensor):
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                return super().forward(*args, **kwargs)

        def convert_weight(self, weight, inplace=False, **kwargs):
            """Convert weight for LoRA patching - dequantize FP8."""
            if isinstance(weight, QuantizedTensor):
                return weight.dequantize()

            if (
                weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
                and self.scale_weight is not None
            ):
                return self._dequantize_weight(weight, self.scale_weight, torch.float32)

            return weight

        def set_weight(
            self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs
        ):
            """Set weight after LoRA patching."""
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
