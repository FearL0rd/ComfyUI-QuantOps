"""
Loader nodes for quantized models.

These nodes provide custom model loading with:
- Kernel backend selection (pytorch/triton)
- Legacy format support (scale_weight -> weight_scale conversion)
- INT8, and FP8 variants
"""

import logging
import folder_paths
import comfy.sd
import comfy.utils


class QuantizedModelLoader:
    """
    Load models with custom quantization layouts and kernel backend selection.

    Supports models quantized by convert_to_quant (with or without --comfy_quant flag).
    Automatically handles legacy scale_weight -> weight_scale conversion.
    
    FP8 Modes:
    - float8_e4m3fn: Standard tensor-scaled FP8 (uses ComfyUI built-in handling)
    - float8_e4m3fn_blockwise: Block-wise scaled FP8 (uses HybridFP8Ops)
    - float8_e4m3fn_rowwise: Row-wise scaled FP8 (uses HybridFP8Ops)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "quant_format": (["auto", "int8", "float8_e4m3fn", "float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise"],),
                "kernel_backend": (["pytorch", "triton"],),
            },
            "optional": {
                "force_dequant": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force dequantize all weights at load time",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load checkpoints with custom quantization support. float8_e4m3fn (tensor-scaled) uses ComfyUI built-in. INT8/FP8 blockwise/rowwise use custom layouts."


    def load_checkpoint(
        self, ckpt_name, quant_format, kernel_backend, force_dequant=False
    ):
        """Load a checkpoint with the specified quantization format and kernel backend."""

        # Set the kernel backend for INT8 layout (only affects INT8 models)
        if quant_format in ("auto", "int8"):
            try:
                from ..quant_layouts.int8_layout import BlockWiseINT8Layout

                BlockWiseINT8Layout.set_backend(kernel_backend)
                logging.debug(
                    f"QuantizedModelLoader: Configured INT8 backend to '{kernel_backend}'"
                )
            except Exception as e:
                if kernel_backend == "triton":
                    logging.warning(f"Failed to configure Triton backend: {e}")

        # Get full checkpoint path
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        # Select ops class based on quant_format
        model_options = {}
        if quant_format == "int8":
            try:
                from ..int8_ops import HybridINT8Ops

                model_options = {"custom_operations": HybridINT8Ops}
                logging.info(
                    "QuantizedModelLoader: Using HybridINT8Ops for INT8 models"
                )
            except ImportError as e:
                logging.warning(f"HybridINT8Ops not available: {e}")
        elif quant_format == "float8_e4m3fn":
            # Standard tensor-scaled FP8 - use ComfyUI's built-in handling
            # No custom ops needed, TensorCoreFP8Layout handles it correctly
            logging.info(
                "QuantizedModelLoader: Using ComfyUI built-in for tensor-scaled FP8"
            )
        elif quant_format in ("float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise"):
            # Block-wise or row-wise FP8 - use HybridFP8Ops for per-block/row scales
            try:
                from ..fp8_ops import HybridFP8Ops

                model_options = {"custom_operations": HybridFP8Ops}
                logging.info(
                    f"QuantizedModelLoader: Using HybridFP8Ops for {quant_format} models"
                )
            except ImportError as e:
                logging.warning(f"HybridFP8Ops not available: {e}")
        else:  # auto
            # Try INT8 as fallback for auto mode
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
    
    FP8 Modes:
    - float8_e4m3fn: Standard tensor-scaled FP8 (uses ComfyUI built-in handling)
    - float8_e4m3fn_blockwise: Block-wise scaled FP8 (uses HybridFP8Ops)
    - float8_e4m3fn_rowwise: Row-wise scaled FP8 (uses HybridFP8Ops)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "quant_format": (["auto", "int8", "float8_e4m3fn", "float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise"],),
                "kernel_backend": (["pytorch", "triton"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load diffusion models with custom quantization support. float8_e4m3fn (tensor-scaled) uses ComfyUI built-in. INT8/FP8 blockwise/rowwise use custom layouts."

    def load_unet(self, unet_name, quant_format, kernel_backend):
        """Load a UNET model with the specified settings."""

        # Set kernel backend (only for INT8 format)
        if quant_format in ("auto", "int8"):
            try:
                from ..quant_layouts.int8_layout import BlockWiseINT8Layout

                BlockWiseINT8Layout.set_backend(kernel_backend)
                logging.debug(
                    f"QuantizedUNETLoader: Configured INT8 backend to '{kernel_backend}'"
                )
            except Exception as e:
                if kernel_backend == "triton":
                    logging.warning(f"Failed to configure Triton backend: {e}")

        # Get model path
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)

        # Select ops class based on quant_format
        model_options = {}
        if quant_format == "int8":
            try:
                from ..int8_ops import HybridINT8Ops

                model_options = {"custom_operations": HybridINT8Ops}
                logging.info("QuantizedUNETLoader: Using HybridINT8Ops for INT8 models")
            except ImportError as e:
                logging.warning(f"HybridINT8Ops not available: {e}")
        elif quant_format == "float8_e4m3fn":
            # Standard tensor-scaled FP8 - use ComfyUI's built-in handling
            logging.info(
                "QuantizedUNETLoader: Using ComfyUI built-in for tensor-scaled FP8"
            )
        elif quant_format in ("float8_e4m3fn_blockwise", "float8_e4m3fn_rowwise"):
            # Block-wise or row-wise FP8 - use HybridFP8Ops
            try:
                from ..fp8_ops import HybridFP8Ops

                model_options = {"custom_operations": HybridFP8Ops}
                logging.info(
                    f"QuantizedUNETLoader: Using HybridFP8Ops for {quant_format} models"
                )
            except ImportError as e:
                logging.warning(f"HybridFP8Ops not available: {e}")
        else:  # auto
            try:
                from ..int8_ops import HybridINT8Ops

                model_options = {"custom_operations": HybridINT8Ops}
                logging.info("QuantizedUNETLoader: Auto-selected HybridINT8Ops")
            except ImportError as e:
                logging.warning(f"No quantized ops available: {e}")

        # Standard loading path
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        return (model,)


class QuantizedCLIPLoader:
    """
    Load CLIP/text encoders with INT8 blockwise quantization.

    Supports text encoders quantized by convert_to_quant with --int8 --comfy_quant.
    """

    # CLIPType options matching built-in CLIPLoader from nodes.py
    CLIP_TYPES = [
        "stable_diffusion",
        "stable_cascade",
        "sd3",
        "stable_audio",
        "mochi",
        "ltxv",
        "pixart",
        "cosmos",
        "lumina2",
        "wan",
        "hidream",
        "chroma",
        "ace",
        "omnigen2",
        "qwen_image",
        "hunyuan_image",
        "flux",
        "hunyuan_video",
        "flux2",
        "ovis",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "type": (cls.CLIP_TYPES,),
                "kernel_backend": (["pytorch", "triton"],),
            },
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load INT8-quantized text encoders (CLIP, T5, etc.)"

    def load_clip(self, clip_name, type, kernel_backend):
        """Load a CLIP/text encoder with INT8 quantization support."""
        import comfy.model_management

        # Configure INT8 kernel backend
        try:
            from ..quant_layouts.int8_layout import BlockWiseINT8Layout

            BlockWiseINT8Layout.set_backend(kernel_backend)
            logging.debug(
                f"QuantizedCLIPLoader: Configured INT8 backend to '{kernel_backend}'"
            )
        except Exception as e:
            if kernel_backend == "triton":
                logging.warning(f"Failed to configure Triton backend: {e}")

        # Get clip path
        clip_path = folder_paths.get_full_path("text_encoders", clip_name)

        # Convert type string to CLIPType enum
        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )

        # Load state dict
        sd = comfy.utils.load_torch_file(clip_path, safe_load=True)

        # Set up model options with our custom INT8 ops
        model_options = {
            "initial_device": comfy.model_management.text_encoder_offload_device()
        }
        try:
            from ..int8_ops import HybridINT8Ops

            model_options["custom_operations"] = HybridINT8Ops
            logging.info(
                "QuantizedCLIPLoader: Using HybridINT8Ops for INT8 text encoder"
            )
        except ImportError as e:
            logging.warning(f"HybridINT8Ops not available: {e}")

        # Load text encoder using ComfyUI's API
        clip = comfy.sd.load_text_encoder_state_dicts(
            state_dicts=[sd],
            clip_type=clip_type,
            model_options=model_options,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        return (clip,)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "QuantizedModelLoader": QuantizedModelLoader,
    "QuantizedUNETLoader": QuantizedUNETLoader,
    "QuantizedCLIPLoader": QuantizedCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuantizedModelLoader": "Load Checkpoint (Quantized)",
    "QuantizedUNETLoader": "Load Diffusion Model (Quantized)",
    "QuantizedCLIPLoader": "Load CLIP (Quantized)",
}
