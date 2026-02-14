

## Installation instructions MXFP8 Support (Blackwell)


## torch 2.10 cuda 13

- required for scaled_mm_v2

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 -U
```

### triton:
```bash
pip install triton pytorch-triton
```


## Install custom comfy-kitchen wheel into your ComfyUI installation

### Build instructions:

- Will require CUDA Toolkit 13.0 installed and added to you environment path

```bash
pip install git+https://github.com/silveroxides/comfy-kitchen.git@fix/int8-tensor-backend#egg=comfy-kitchen --no-deps --force-reinstall --no-cache-dir --no-build-isolation
```

- OR

```bash
git clone https://github.com/silveroxides/comfy-kitchen
cd comfy-kitchen
git checkout fix/int8-tensor-backend
python setup.py build_ext bdist_wheel
```

### Regular venv installation:

- Download wheel from https://huggingface.co/silveroxides/Chroma1-HD-fp8-scaled/resolve/main/experimental/comfy_kitchen-0.2.7-cp312-abi3-linux_x86_64.whl
- Copy the .whl file to your input directory inside ComfyUI directory
- Open terminal in your ComfyUI directory
- The .whl in this repo works for Python 3.12 and 3.13.

```bash
source venv/bin/activate
pip install input/comfy_kitchen-0.2.7-cp312-abi3-linux_x86_64.whl --no-deps --force-reinstall --no-cache-dir
```

## ComfyUI-QuantOps

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/silveroxides/ComfyUI-QuantOps
```

## OPTIONAL

### convert_to_quant

- quantization tool

```bash
pip install convert-to-quant -U --no-deps --force-reinstall --no-cache-dir
```

- example command:

```bash
ctq -i ./flux2-dev.safetensors -o ./flux2-dev-mxfp8mixed.safetensors --mxfp8 --comfy_quant --low-memory --save-quant-metadata --simple
```

- Use the help for general command info and filters help for specific models filers

```bash
ctq -h
ctq -hf
```