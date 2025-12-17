# ComfyUI-QuantOps

Extended quantization layouts for ComfyUI, enabling loading and inference with models quantized by [convert_to_quant](https://github.com/silveroxides/convert_to_quant).

## Supported Formats

| Format | Layout | Description |
|--------|--------|-------------|
| FP8 (tensor-wise) | `TensorCoreFP8Layout` | Standard FP8 with tensor-wise scaling |
| FP8 (row-wise) | `RowWiseFP8Layout` | Per-row FP8 scaling |
| FP8 (block-wise) | `BlockWiseFP8Layout` | 2D block-wise FP8 scaling |
| INT8 (block-wise) | `BlockWiseINT8Layout` | Block-wise INT8 with Triton kernels |
| NF4 | `NF4Layout` | 4-bit normal float (bitsandbytes-compatible) |
| FP4 | `FP4Layout` | 4-bit floating point |

## Installation

1. Clone to your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/silveroxides/ComfyUI-QuantOps.git
   ```

2. (Optional) Install Triton for GPU-accelerated INT8:
   ```bash
   # Activate your ComfyUI venv first!
   # Linux
   pip install triton
   # Windows
   pip install triton-windows
   ```

## Usage

Use the **QuantizedModelLoader** node to load models created by `convert_to_quant`:

1. Quantize your model with convert_to_quant:
   ```bash
   convert_to_quant -i model.safetensors --int8 --comfy_quant
   ```

2. Place the output in your ComfyUI models/checkpoints folder

3. Use QuantizedModelLoader node with:
   - `quant_format`: Select the format matching your model
   - `kernel_backend`: Choose "pytorch" (always works) or "triton" (faster, requires installation)

## License

MIT License
