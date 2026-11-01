# Third-party software and models

Pixforge is distributed together with, or downloads, the components below. Every entry
lists its purpose and license. Only permissive licenses are used in the default setup.

## Models (weights)

| Model | Used for | Source | License | Shipped how |
|---|---|---|---|---|
| Real_HAT_GAN_SRx4, Real_HAT_GAN_sharper | „Fotos (Standard)“, „Fotos, schärfer“ | [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT) | Apache-2.0 | converted to ONNX, downloaded on demand |
| HAT_SRx2/3/4 (ImageNet-pretrain), HAT-L_SRx2/4 | „Saubere Bilder“, „Maximale Qualität“ | XPixelGroup/HAT | Apache-2.0 | converted to ONNX, downloaded on demand |
| SCUNet (color, real PSNR) | noise reduction before upscaling | [cszn/SCUNet](https://github.com/cszn/SCUNet) | Apache-2.0 | evaluated in stage 1 |
| NAFNet-SIDD-width64 | noise reduction (comparison) | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | MIT | evaluated in stage 1 |
| FBCNN (color) | JPEG artifact removal | [jiaxi-jiang/FBCNN](https://github.com/jiaxi-jiang/FBCNN) | Apache-2.0 | evaluated in stage 1 |

Architecture code adapted from these repositories lives in
`backend/src/pixforge/models/*_arch.py`; each file header names its origin and license.
HAT and SCUNet contain parts derived from SwinIR (Apache-2.0) and timm (Apache-2.0).

## Python runtime dependencies

| Package | Purpose | License |
|---|---|---|
| numpy | arrays | BSD-3-Clause |
| Pillow | image I/O | MIT-CMU (HPND) |
| opencv-python-headless | resampling, classic filters | Apache-2.0 (OpenCV), MIT (wheel packaging) |
| onnxruntime-directml | inference on DirectML / CPU | MIT |
| truststore | use the Windows certificate store for downloads | MIT |

## Developer-only dependencies (not shipped)

| Package | Purpose | License |
|---|---|---|
| torch (CPU) | loading checkpoints, ONNX export | BSD-3-Clause |
| onnx, onnxscript | ONNX export / checks | Apache-2.0 / MIT |
| gdown | downloading upstream weights from Google Drive | MIT |
| pytest, ruff | tests, lint | MIT |
