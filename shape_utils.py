"""
Shape restoration utilities for 4-bit quantized models.

ComfyUI's model_detection.py reads tensor shapes to detect model architecture.
4-bit packed weights have different shapes than originals, causing detection failure.

This module provides utilities to:
1. Parse quant_state metadata to get original shapes
2. Create proxy tensors with correct shapes for model detection
3. Restore original packed weights after detection
"""

import torch
import logging

# Use the same tensor parsing as the rest of the codebase
from .comfy_quant_helpers import tensor_to_dict


def parse_quant_state(quant_state_tensor):
    """Parse quant_state tensor (uint8 JSON) to dict using comfy_quant_helpers."""
    try:
        return tensor_to_dict(quant_state_tensor)
    except Exception as e:
        logging.debug(f"Failed to parse quant_state: {e}")
        return None


class ShapeProxy:
    """
    Lightweight proxy that mimics a tensor for model_detection purposes.
    
    Only provides .shape, .dtype, .device attributes that model_detection reads.
    Uses no memory for actual tensor data.
    """
    def __init__(self, shape, dtype=torch.bfloat16):
        self._shape = torch.Size(shape) if not isinstance(shape, torch.Size) else shape
        self._dtype = dtype
        
    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def device(self):
        return torch.device('cpu')
    
    def __repr__(self):
        return f"ShapeProxy(shape={self._shape}, dtype={self._dtype})"


def restore_shapes_for_detection(state_dict, prefix=""):
    """
    Pre-process state_dict to restore original shapes for model detection.
    
    For each 4-bit quantized weight, replaces the packed uint8 tensor with a 
    ShapeProxy that has the original shape. This allows model_detection
    to work correctly without allocating memory for proxy tensors.
    
    Returns:
        tuple: (processed_sd, shape_info) where shape_info maps weight keys to their
               packed tensors for later restoration
    """
    shape_info = {}
    processed_sd = {}
    
    # Find all quant_state tensors with shape info
    quant_states = {}
    
    # Debug: log all keys to understand structure
    quant_state_keys = [k for k in state_dict.keys() if 'quant_state' in k.lower() or 'comfy_quant' in k.lower()]
    if quant_state_keys:
        logging.info(f"Shape restoration: Found {len(quant_state_keys)} quant_state/comfy_quant keys")
        logging.debug(f"Sample keys: {quant_state_keys[:5]}")
    
    for key in list(state_dict.keys()):
        # Check for legacy bitsandbytes format
        if '.quant_state.bitsandbytes__' in key:
            # Extract layer prefix
            layer_prefix = key.split('.quant_state.')[0]
            try:
                quant_state = parse_quant_state(state_dict[key])
                if quant_state:
                    logging.debug(f"Parsed quant_state for {layer_prefix}: keys={list(quant_state.keys())}")
                    if 'shape' in quant_state:
                        # Validate shape is list/tuple of ints
                        shape = quant_state['shape']
                        if isinstance(shape, (list, tuple)) and len(shape) >= 1:
                            quant_states[layer_prefix] = quant_state
                            logging.debug(f"Found shape {shape} for {layer_prefix}")
            except Exception as e:
                logging.debug(f"Failed to parse quant_state {key}: {e}")
    
    # Also check comfy_quant for shape info
    for key in list(state_dict.keys()):
        if key.endswith('.comfy_quant'):
            layer_prefix = key[:-len('.comfy_quant')]
            try:
                comfy_quant = parse_quant_state(state_dict[key])
                if comfy_quant:
                    logging.debug(f"Parsed comfy_quant for {layer_prefix}: keys={list(comfy_quant.keys())}")
                    if 'shape' in comfy_quant:
                        shape = comfy_quant['shape']
                        if isinstance(shape, (list, tuple)) and len(shape) >= 1:
                            quant_states[layer_prefix] = comfy_quant
                            logging.debug(f"Found shape {shape} for {layer_prefix} (comfy_quant)")
            except Exception as e:
                logging.debug(f"Failed to parse comfy_quant {key}: {e}")
    
    logging.info(f"Shape restoration: Found {len(quant_states)} layers with valid shape info")
    
    # Process all keys
    for key in state_dict.keys():
        value = state_dict[key]
        
        # Check if this is a packed weight that needs shape restoration
        if key.endswith('.weight') and value.dtype == torch.uint8:
            layer_prefix = key[:-len('.weight')]
            if layer_prefix in quant_states:
                orig_shape = quant_states[layer_prefix].get('shape')
                if orig_shape:
                    try:
                        # Validate shape is proper format
                        if not isinstance(orig_shape, (list, tuple)):
                            logging.warning(f"Invalid shape type for {key}: {type(orig_shape)}")
                            processed_sd[key] = value
                            continue
                        if len(orig_shape) < 1:
                            logging.warning(f"Empty shape for {key}")
                            processed_sd[key] = value
                            continue
                        
                        # Store original packed weight for later
                        shape_info[key] = value
                        # Create lightweight shape proxy (no memory allocation)
                        processed_sd[key] = ShapeProxy(orig_shape)
                        logging.debug(f"Shape restoration: {key} {value.shape} -> {orig_shape}")
                        continue
                    except Exception as e:
                        logging.warning(f"Failed to create proxy for {key} with shape {orig_shape}: {e}")
        
        processed_sd[key] = value
    
    return processed_sd, shape_info


def restore_packed_weights(state_dict, shape_info):
    """
    Restore original packed weights after model detection.
    
    Replaces proxy tensors with actual packed weights from shape_info.
    """
    for key, packed_weight in shape_info.items():
        if key in state_dict:
            state_dict[key] = packed_weight
            logging.debug(f"Restored packed weight: {key}")
    return state_dict


def load_4bit_safetensors(path):
    """
    Load a 4-bit quantized safetensors file with shape-aware preprocessing.
    
    Returns:
        tuple: (state_dict, shape_info, metadata) 
               where shape_info can be used to restore packed weights
    """
    import safetensors.torch
    
    # Load raw state dict
    with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    # Pre-process for model detection
    processed_sd, shape_info = restore_shapes_for_detection(state_dict)
    
    return processed_sd, shape_info, metadata
