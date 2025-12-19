"""
Loader nodes for quantized models.

These nodes provide custom model loading with:
- Kernel backend selection (pytorch/triton)
- Legacy format support (scale_weight -> weight_scale conversion)
- Support for INT8, NF4, FP4, and FP8 variants
"""

import os
import logging
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
        
        # Load the model with our custom operations
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
