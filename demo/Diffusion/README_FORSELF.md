# Introduction

This demo application ("demoDiffusion") showcases the acceleration of SDXL-Lightning pipeline using TensorRT.

# Setup

### Clone the TensorRT OSS repository

```bash
git clone git@github.com:LiaoFJ/TensorRT.git -b release/9.2 --single-branch
cd TensorRT
```

### Launch NVIDIA pytorch container

Install nvidia-docker using [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```bash
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/pytorch:23.12-py3 /bin/bash
```

### Install latest TensorRT release

```bash
python3 -m pip install tensorrt==8.6.1.post1
```

Check your installed version using:
`python3 -c 'import tensorrt;print(tensorrt.__version__)'`

### Install required packages

```bash
export TRT_OSSPATH=/workspace
cd $TRT_OSSPATH/demo/Diffusion
pip3 install -r requirements.txt
```

# Running demoDiffusion with SDXL-Lightning


### Generate an image with Stable Diffusion XL - Lightning guided by a single text prompt

Run the below command to generate an image with Stable Diffusion XL - Lightning. Please noted that 8-step sampling is used as default when generation.

```bash
python3 demo_txt2img_xl.py "Einstein" --version xl-1.0 --onnx-dir onnx-sdxl-lightning --engine-dir engine-sdxl-lightning --denoising-steps 8 --scheduler Lightning --guidance-scale 0.0
```
当前的默认值是sdxl-lightning, 在get_path()里面extension对应的路径是sdxl-lightning的。同时，unetxl的加载部分也是extension对应lightning
同时，默认值里面，guidance_scale设置为0， 因此也没有batch=2X

## Configuration options
- Noise scheduler can be set using `--scheduler <scheduler>`. Note: not all schedulers are available for every version.
- To accelerate engine building time use `--timing-cache <path to cache file>`. The cache file will be created if it does not already exist. Note that performance may degrade if cache files are used across multiple GPU targets. It is recommended to use timing caches only during development. To achieve the best perfromance in deployment, please build engines without timing cache.
- Specify new directories for storing onnx and engine files when switching between versions, LoRAs, ControlNets, etc. This can be done using `--onnx-dir <new onnx dir>` and `--engine-dir <new engine dir>`.
- Inference performance can be improved by enabling [CUDA graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) using `--use-cuda-graph`. Enabling CUDA graphs requires fixed input shapes, so this flag must be combined with `--build-static-batch` and cannot be combined with `--build-dynamic-shape`.

三个仓库：local debug, 9.2 以及dev_debug: 9.2 是用来做远程编译的（正式版）；local debug是来测试新功能的；dev_debug是来测试正式版的一些衍生问题的。
