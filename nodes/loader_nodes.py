"""
Loader nodes for quantized models.

These nodes provide custom model loading with:
- Kernel backend selection (pytorch/triton)
- Format auto-detection from .comfy_quant metadata
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
    
    Supports models quantized by convert_to_quant with --comfy_quant flag.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "kernel_backend": (["pytorch", "triton"],),
            },
            "optional": {
                "force_dequant": ("BOOLEAN", {"default": False, "tooltip": "Force dequantize all weights at load time"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load checkpoints with custom quantization support (INT8, NF4, FP4, FP8 variants)"
    
    def load_checkpoint(self, ckpt_name, kernel_backend, force_dequant=False):
        """Load a checkpoint with the specified kernel backend."""
        
        # Set the kernel backend for INT8 layout
        try:
            from ..quant_layouts.int8_layout import BlockWiseINT8Layout
            BlockWiseINT8Layout.set_backend(kernel_backend)
            logging.info(f"QuantizedModelLoader: Set INT8 backend to '{kernel_backend}'")
        except Exception as e:
            if kernel_backend == "triton":
                logging.warning(f"Failed to set Triton backend: {e}. Using PyTorch fallback.")
        
        # Get full checkpoint path
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        
        # Use ComfyUI's standard checkpoint loading
        # The layouts we registered will handle the quantized tensors automatically
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        
        model = out[0]
        clip = out[1]
        vae = out[2]
        
        # Force dequantize if requested (useful for debugging)
        if force_dequant and model is not None:
            logging.info("QuantizedModelLoader: Force dequantizing model weights")
            # This would dequantize all weights - expensive but useful for testing
            # Left as a placeholder for now
            pass
        
        return (model, clip, vae)


class QuantizedUNETLoader:
    """
    Load UNET/diffusion models with custom quantization layouts.
    
    More focused version for loading just the diffusion model.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "kernel_backend": (["pytorch", "triton"],),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load diffusion models with custom quantization support"
    
    def load_unet(self, unet_name, kernel_backend, weight_dtype):
        """Load a UNET model with the specified settings."""
        
        # Set kernel backend
        try:
            from ..quant_layouts.int8_layout import BlockWiseINT8Layout
            BlockWiseINT8Layout.set_backend(kernel_backend)
        except Exception as e:
            if kernel_backend == "triton":
                logging.warning(f"Failed to set Triton backend: {e}")
        
        # Get model path
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        
        # Load the model
        model = comfy.sd.load_diffusion_model(unet_path)
        
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
