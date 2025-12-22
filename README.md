# ComfyUI-QuantOps

Extended quantization layouts for ComfyUI, enabling loading and inference with models quantized by [convert_to_quant](https://github.com/silveroxides/convert_to_quant).

## Supported Formats

| Format | Layout | Description |
|--------|--------|-------------|
| FP8 (tensor-wise) | `TensorCoreFP8Layout` | Standard FP8 with tensor-wise scaling | Supported |
| FP8 (row-wise) | `RowWiseFP8Layout` | Per-row FP8 scaling | WIP |
| FP8 (block-wise) | `BlockWiseFP8Layout` | 2D block-wise FP8 scaling | WIP |
| INT8 (block-wise) | `BlockWiseINT8Layout` | Block-wise INT8 with Triton kernels | Supported |

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

1. Quantize your model with [convert_to_quant](https://github.com/silveroxides/convert_to_quant):
   ```bash
   convert_to_quant -i model.safetensors --int8 --comfy_quant --simple --block_size 128
   ```

2. Place the output in your ComfyUI models/checkpoints folder


## License

MIT License

## Acknowledgements

- [lyogavin](https://github.com/lyogavin) for [PR #10864](https://github.com/comfyanonymous/ComfyUI/pull/10864) to ComfyUI.
- [Clybius](https://github.com/Clybius) for inspiring me to take on quantization and his [Learned-Rounding](https://github.com/Clybius/Learned-Rounding) repository.
