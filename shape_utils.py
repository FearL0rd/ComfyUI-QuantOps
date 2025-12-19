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
import json
import logging


def parse_quant_state(quant_state_tensor):
    """Parse quant_state tensor (uint8 JSON) to dict."""
    try:
        return json.loads(quant_state_tensor.numpy().tobytes())
    except Exception as e:
        logging.debug(f"Failed to parse quant_state: {e}")
        return None


def restore_shapes_for_detection(state_dict, prefix=""):
    """
    Pre-process state_dict to restore original shapes for model detection.
    
    For each 4-bit quantized weight, replaces the packed uint8 tensor with a 
    proxy bfloat16 tensor of the original shape. This allows model_detection
    to work correctly.
    
    Returns:
        tuple: (processed_sd, shape_info) where shape_info maps weight keys to their
               packed tensors for later restoration
    """
    shape_info = {}
    processed_sd = {}
    
    # Find all quant_state tensors with shape info
    quant_states = {}
    for key in list(state_dict.keys()):
        if '.quant_state.bitsandbytes__' in key or key.endswith('.quant_state.bitsandbytes__nf4') or key.endswith('.quant_state.bitsandbytes__fp4') or key.endswith('.quant_state.bitsandbytes__af4'):
            # Extract layer prefix
            layer_prefix = key.split('.quant_state.')[0]
            quant_state = parse_quant_state(state_dict[key])
            if quant_state and 'shape' in quant_state:
                quant_states[layer_prefix] = quant_state
    
    # Also check comfy_quant for shape info
    for key in list(state_dict.keys()):
        if key.endswith('.comfy_quant'):
            layer_prefix = key[:-len('.comfy_quant')]
            try:
                comfy_quant = parse_quant_state(state_dict[key])
                if comfy_quant and 'shape' in comfy_quant:
                    quant_states[layer_prefix] = comfy_quant
            except:
                pass
    
    # Process all keys
    for key in state_dict.keys():
        value = state_dict[key]
        
        # Check if this is a packed weight that needs shape restoration
        if key.endswith('.weight') and value.dtype == torch.uint8:
            layer_prefix = key[:-len('.weight')]
            if layer_prefix in quant_states:
                orig_shape = quant_states[layer_prefix].get('shape')
                if orig_shape:
                    # Store original packed weight for later
                    shape_info[key] = value
                    # Create proxy tensor with original shape
                    proxy = torch.zeros(orig_shape, dtype=torch.bfloat16)
                    processed_sd[key] = proxy
                    logging.debug(f"Shape restoration: {key} {value.shape} -> {orig_shape}")
                    continue
        
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
