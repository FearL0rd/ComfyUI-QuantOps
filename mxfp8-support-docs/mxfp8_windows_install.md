# Installation instructions MXFP8 Support (Blackwell) Windows Wheel


## torch 2.10 cuda 13

- required for scaled_mm_v2

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 -U
```

## Install custom comfy-kitchen wheel into your ComfyUI installation

### Regular venv installation:

- Download wheel from https://huggingface.co/silveroxides/Chroma1-HD-fp8-scaled/resolve/main/experimental/comfy_kitchen-0.2.7-cp312-abi3-win_amd64.whl
- Copy the .whl file to your input directory inside ComfyUI directory
- Open Terminal in your ComfyUI directory

```bash
.\venv\Scripts\Activate.ps1
pip install .\input\comfy_kitchen-0.2.7-cp312-abi3-win_amd64.whl --no-deps --force-reinstall --no-cache-dir
```

### ComfyUI portable installation:

- Copy the .whl file to the same folder you have `run_cpu.bat` and the `run_nvidia_gpu.bat` files.
- Right click in that folder in Windows Explorer and select `Open in Terminal`
- Paste the following command and press Enter

```bash
.\python_embeded\python.exe -s -m pip install .\comfy_kitchen-0.2.7-cp312-abi3-win_amd64.whl --no-deps --force-reinstall --no-cache-dir
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