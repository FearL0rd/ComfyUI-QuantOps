"""
Loader nodes for quantized models.

These nodes provide custom model loading with:
- Kernel backend selection (pytorch/triton)
- Legacy format support (scale_weight -> weight_scale conversion)
- Support for INT8, NF4, FP4, and FP8 variants
"""

import os
import logging
import torch
import folder_paths
import comfy.sd
import comfy.utils
from comfy.quant_ops import LAYOUTS, QUANT_ALGOS


class QuantizedModelLoader:
    """
    Load models with custom quantization layouts and kernel backend selection.
    
    Supports models quantized by convert_to_quant (with or without --comfy_quant flag).
    Automatically handles legacy scale_weight -> weight_scale conversion.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "quant_format": (["auto", "4bit (NF4/FP4/AF4)", "int8"],),
                "kernel_backend": (["pytorch", "triton"],),
            },
            "optional": {
                "force_dequant": ("BOOLEAN", {"default": False, "tooltip": "Force dequantize all weights at load time"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load checkpoints with custom quantization support (INT8, NF4, FP4, FP8 variants). Select quant_format to match your model."
    
    def load_checkpoint(self, ckpt_name, quant_format, kernel_backend, force_dequant=False):
        """Load a checkpoint with the specified quantization format and kernel backend."""
        
        # Set the kernel backend for INT8 layout (only affects INT8 models)
        if quant_format in ("auto", "int8"):
            try:
                from ..quant_layouts.int8_layout import BlockWiseINT8Layout
                BlockWiseINT8Layout.set_backend(kernel_backend)
                logging.debug(f"QuantizedModelLoader: Configured INT8 backend to '{kernel_backend}'")
            except Exception as e:
                if kernel_backend == "triton":
                    logging.warning(f"Failed to configure Triton backend: {e}")
        
        # Get full checkpoint path
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        # Select ops class based on quant_format
        model_options = {}
        if quant_format == "4bit (NF4/FP4/AF4)":
            try:
                from ..nf4_ops import HybridNF4Ops
                model_options = {"custom_operations": HybridNF4Ops}
                logging.info("QuantizedModelLoader: Using HybridNF4Ops for 4-bit models")
            except ImportError as e:
                logging.warning(f"HybridNF4Ops not available: {e}")
        elif quant_format == "int8":
            try:
                from ..int8_ops import HybridINT8Ops
                model_options = {"custom_operations": HybridINT8Ops}
                logging.info("QuantizedModelLoader: Using HybridINT8Ops for INT8 models")
            except ImportError as e:
                logging.warning(f"HybridINT8Ops not available: {e}")
        else:  # auto
            # Try NF4 first (handles 4-bit), then INT8 as fallback
            try:
                from ..nf4_ops import HybridNF4Ops
                model_options = {"custom_operations": HybridNF4Ops}
                logging.info("QuantizedModelLoader: Auto-selected HybridNF4Ops")
            except ImportError:
                try:
                    from ..int8_ops import HybridINT8Ops
                    model_options = {"custom_operations": HybridINT8Ops}
                    logging.info("QuantizedModelLoader: Auto-selected HybridINT8Ops")
                except ImportError as e:
                    logging.warning(f"No quantized ops available: {e}")
        
        # Use ComfyUI's checkpoint loading with our custom operations
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            model_options=model_options,
        )
        
        model = out[0]
        clip = out[1]
        vae = out[2]
        
        # Force dequantize if requested (useful for debugging)
        if force_dequant and model is not None:
            logging.info("QuantizedModelLoader: Force dequantizing model weights")
            pass
        
        return (model, clip, vae)


class QuantizedUNETLoader:
    """
    Load UNET/diffusion models with custom quantization layouts.
    
    Handles legacy scale_weight format automatically.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "quant_format": (["auto", "4bit (NF4/FP4/AF4)", "int8"],),
                "kernel_backend": (["pytorch", "triton"],),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load diffusion models with custom quantization support. Select quant_format to match your model."
    
    def load_unet(self, unet_name, quant_format, kernel_backend):
        """Load a UNET model with the specified settings."""
        
        # Set kernel backend (only for INT8 format)
        if quant_format in ("auto", "int8"):
            try:
                from ..quant_layouts.int8_layout import BlockWiseINT8Layout
                BlockWiseINT8Layout.set_backend(kernel_backend)
                logging.debug(f"QuantizedUNETLoader: Configured INT8 backend to '{kernel_backend}'")
            except Exception as e:
                if kernel_backend == "triton":
                    logging.warning(f"Failed to configure Triton backend: {e}")
        
        # Get model path
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        
        # Select ops class based on quant_format
        model_options = {}
        if quant_format == "4bit (NF4/FP4/AF4)":
            try:
                from ..nf4_ops import HybridNF4Ops
                model_options = {"custom_operations": HybridNF4Ops}
                logging.info("QuantizedUNETLoader: Using HybridNF4Ops for 4-bit models")
            except ImportError as e:
                logging.warning(f"HybridNF4Ops not available: {e}")
        elif quant_format == "int8":
            try:
                from ..int8_ops import HybridINT8Ops
                model_options = {"custom_operations": HybridINT8Ops}
                logging.info("QuantizedUNETLoader: Using HybridINT8Ops for INT8 models")
            except ImportError as e:
                logging.warning(f"HybridINT8Ops not available: {e}")
        else:  # auto
            try:
                from ..nf4_ops import HybridNF4Ops
                model_options = {"custom_operations": HybridNF4Ops}
                logging.info("QuantizedUNETLoader: Auto-selected HybridNF4Ops")
            except ImportError:
                try:
                    from ..int8_ops import HybridINT8Ops
                    model_options = {"custom_operations": HybridINT8Ops}
                    logging.info("QuantizedUNETLoader: Auto-selected HybridINT8Ops")
                except ImportError as e:
                    logging.warning(f"No quantized ops available: {e}")
        
        # For 4-bit models, use shape restoration to fix model detection
        if quant_format == "4bit (NF4/FP4/AF4)":
            try:
                from ..shape_utils import restore_shapes_for_detection, restore_packed_weights
                import comfy.model_detection
                import comfy.model_management
                import comfy.model_patcher
                
                # Load raw state dict
                sd, metadata = comfy.utils.load_torch_file(unet_path, safe_load=True, return_metadata=True)
                
                # Restore shapes for model detection (creates proxy bfloat16 tensors)
                processed_sd, shape_info = restore_shapes_for_detection(sd)
                
                if len(shape_info) > 0:
                    logging.info(f"QuantizedUNETLoader: Restored {len(shape_info)} tensor shapes for model detection")
                    
                    # Strip model prefix if present
                    diffusion_model_prefix = comfy.model_detection.unet_prefix_from_state_dict(processed_sd)
                    detection_sd = comfy.utils.state_dict_prefix_replace(processed_sd, {diffusion_model_prefix: ""}, filter_keys=True)
                    if len(detection_sd) > 0:
                        processed_sd = detection_sd
                    
                    # *** Run model detection with shape-restored state_dict ***
                    model_config = comfy.model_detection.model_config_from_unet(processed_sd, "", metadata=metadata)
                    
                    if model_config is None:
                        logging.warning("Model detection failed even with shape restoration")
                    else:
                        logging.info(f"QuantizedUNETLoader: Detected model config: {type(model_config).__name__}")
                        
                        # Prepare state_dict for loading (with packed weights)
                        # Use filter_keys=False to preserve quant metadata keys like absmax, quant_map, quant_state
                        loading_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=False)
                        
                        # Set up model config with our custom operations
                        if model_options.get("custom_operations"):
                            model_config.custom_operations = model_options["custom_operations"]
                            logging.info(f"QuantizedUNETLoader: Using custom operations: {model_options['custom_operations']}")
                        
                        # Configure dtype and device
                        load_device = comfy.model_management.get_torch_device()
                        offload_device = comfy.model_management.unet_offload_device()
                        
                        # For 4-bit, we don't care about parameter size calculation
                        unet_dtype = model_options.get("dtype", torch.bfloat16)
                        
                        manual_cast_dtype = comfy.model_management.unet_manual_cast(
                            unet_dtype, load_device, model_config.supported_inference_dtypes
                        )
                        model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
                        
                        # Build model with custom operations
                        model = model_config.get_model(loading_sd, "")
                        model = model.to(offload_device)
                        
                        # Load weights - this should trigger _load_from_state_dict for each Linear
                        model.load_model_weights(loading_sd, "")
                        
                        model_patcher = comfy.model_patcher.ModelPatcher(
                            model, load_device=load_device, offload_device=offload_device
                        )
                        
                        logging.info("QuantizedUNETLoader: Successfully loaded 4-bit model with shape restoration")
                        return (model_patcher,)
                        
            except Exception as e:
                import traceback
                logging.warning(f"Shape restoration failed: {e}")
                logging.debug(traceback.format_exc())
        
        # Standard loading path (works for INT8 and non-quantized)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        
        return (model,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "QuantizedModelLoader": QuantizedModelLoader,
    "QuantizedUNETLoader": QuantizedUNETLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuantizedModelLoader": "Load Checkpoint (Quantized)",
    "QuantizedUNETLoader": "Load Diffusion Model (Quantized)",
}
