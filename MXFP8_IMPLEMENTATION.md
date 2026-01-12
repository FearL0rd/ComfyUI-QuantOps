# MXFP8 Implementation Plan for ComfyUI-QuantOps

## Overview

This document outlines the implementation plan for adding MXFP8 (Microscaling FP8) support to ComfyUI-QuantOps, enabling loading and inference of models quantized by `convert_to_quant --mxfp8`.

---

## External Dependencies

### comfy-kitchen Fork

- **Repository**: [silveroxides/comfy-kitchen](https://github.com/silveroxides/comfy-kitchen)
- **Branch**: `sc_mm_mxfp8_sync`
- **Relevant Files**:
  - `comfy_kitchen/tensor/mxfp8.py` - Reference `TensorCoreMXFP8Layout` implementation
  - `comfy_kitchen/scaled_mm_v2.py` - Block-scaled matmul with `ScalingType.BlockWise1x32`
  - `comfy_kitchen/__init__.py` - Exports `quantize_mxfp8`, `dequantize_mxfp8`, `scaled_mm_mxfp8`
  - `samples/mxfp8_model_patcher.py` - Example ComfyUI node for on-the-fly MXFP8

### convert_to_quant

- **Repository**: [silveroxides/convert_to_quant](https://github.com/silveroxides/convert_to_quant)
- **Branch**: `feature/mxfp8-support`
- **Commit**: `d8a2b1c`
- **Output Format**:
  - `.weight` → `float8_e4m3fn` (FP8 E4M3)
  - `.weight_scale` → `uint8` (E8M0 block scales, swizzled layout)
  - `.comfy_quant` → JSON metadata: `{"format": "mxfp8", "group_size": 32, "orig_dtype": "...", "orig_shape": [...]}`

---

## Current State

### What's Been Started

| File | Status | Notes |
|------|--------|-------|
| `quant_layouts/mxfp8_layout.py` | Incomplete | Basic structure, PyTorch fallback dequantize, `ck.dequantize_mxfp8` integration started |
| `quant_layouts/__init__.py` | Modified | Added `TensorCoreMXFP8Layout` export |
| `fp8_ops.py` | Modified | Added `TensorCoreMXFP8Layout` case to loader |

### Issues with Current Implementation

1. **No native matmul** - Currently only dequantizes, causing:
   - Full model dequantization on load → OOM on large models
   - No performance benefit from MXFP8 compression for compute
   
2. **Missing quantize function for activations** - `ck.quantize_mxfp8` needed for runtime input quantization

3. **Node options not implemented** - No ComfyUI node to select/enable MXFP8 ops

4. **ComfyUI loading complexities not addressed**:
   - Model patcher integration
   - Memory management during loading
   - Device placement (offloading)

---

## What Needs To Be Done

### Phase 1: Core Layout Implementation

- [ ] **Rewrite `mxfp8_layout.py`** to match comfy-kitchen reference closely
  - Use `ck.quantize_mxfp8` for activation quantization
  - Use `ck.dequantize_mxfp8` for weight dequantization
  - Use `ck.scaled_mm_mxfp8` for native matmul (Blackwell)
  - Proper E8M0 scale handling (uint8 ↔ float8_e8m0fnu view)

### Phase 2: Operation Handlers

- [ ] **`aten.linear.default`** handler with:
  - Check for `supports_fast_matmul()` (SM >= 10.0)
  - If Blackwell: quantize input on-the-fly + native scaled_mm
  - If pre-Blackwell: lazy dequantize weight only when needed (not OOM)

- [ ] **`aten.mm.default`** handler (similar pattern)

- [ ] **`aten.addmm.default`** handler (bias fusion)

- [ ] **`aten.t.default`** handler (logical transpose flag, no data movement)

### Phase 3: Loader Integration

- [ ] **Update `fp8_ops.py`** HybridFP8Ops.Linear:
  - Detect `format: "mxfp8"` in comfy_quant metadata
  - Create `TensorCoreMXFP8Layout.Params` correctly
  - Handle E8M0 scales (uint8 storage)

- [ ] **Memory-efficient loading**:
  - Don't dequantize entire model on load
  - Keep weights quantized, dequantize per-layer during forward

### Phase 4: ComfyUI Node Integration

- [ ] **Add `HybridMXFP8Ops` class** (or extend `HybridFP8Ops`) in `fp8_ops.py`

- [ ] **Node options** for:
  - Model path selection
  - Force dequantize fallback (for debugging)
  - Memory mode (keep quantized vs dequantize)

- [ ] **README update** with MXFP8 usage instructions

### Phase 5: Testing & Verification

- [ ] Load MXFP8 model without OOM
- [ ] Verify dequantized output matches convert_to_quant output
- [ ] Benchmark memory usage vs FP16/BF16
- [ ] Test on Blackwell GPU (if available) for native matmul

---

## Hardware Requirements

| Feature | Requirement |
|---------|-------------|
| MXFP8 weight storage | Any GPU with PyTorch FP8 support |
| Native MXFP8 matmul | SM >= 10.0 (Blackwell) |
| `scaled_mm` v2 API | PyTorch 2.10+ |

---

## Key comfy-kitchen API

```python
import comfy_kitchen as ck

# Quantize tensor to MXFP8
qdata, block_scale = ck.quantize_mxfp8(tensor, pad_32x=True)

# Dequantize MXFP8 to target dtype
tensor = ck.dequantize_mxfp8(qdata, block_scale, orig_dtype)

# Native block-scaled matmul (Blackwell only)
result = ck.scaled_mm_mxfp8(a_qdata, b_qdata, block_scale_a, block_scale_b, bias, out_dtype)
```

---

## Reference Implementation

The authoritative reference is `comfy-kitchen-temp/comfy_kitchen/tensor/mxfp8.py` which shows:

1. How to handle the `transposed` flag for logical transpose
2. How to slice output to original shape after padded matmul
3. How to delegate to `ck.quantize_mxfp8` / `ck.dequantize_mxfp8` / `ck.scaled_mm_mxfp8`
4. Proper error handling with dequantize fallback
