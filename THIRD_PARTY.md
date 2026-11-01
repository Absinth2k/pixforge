# Third-party software and models

Pixforge is distributed together with, or downloads, the components below. Every entry
lists its purpose and license. Only permissive licenses are used in the default setup; the one
exception (GFPGAN) is an optional download that is marked in the app.

## Models (weights)

All models are converted to ONNX and downloaded on demand from the Pixforge model release.
Each release asset comes with the original license text.

| ID | Used for | Source | License |
|---|---|---|---|
| `photo`, `photo-sharp` | „Fotos (Standard)“, „Fotos, schärfer“ (Real_HAT_GAN_SRx4 / _sharper) | [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT) | Apache-2.0 |
| `clean-x2/x3/x4`, `max-x2/x4` | „Saubere Bilder“, „Maximale Qualität“ (HAT / HAT-L, ImageNet-pretrain) | XPixelGroup/HAT | Apache-2.0 |
| `denoise`, `denoise-gan` | noise reduction (SCUNet color real PSNR / GAN) | [cszn/SCUNet](https://github.com/cszn/SCUNet) | Apache-2.0 |
| `jpeg` | JPEG artifact removal (FBCNN color) | [jiaxi-jiang/FBCNN](https://github.com/jiaxi-jiang/FBCNN) | Apache-2.0 |
| `face` | face restoration, default (RestoreFormer++) | [wzhouxiff/RestoreFormerPlusPlus](https://github.com/wzhouxiff/RestoreFormerPlusPlus) | Apache-2.0 |
| `face-gfpgan` | face restoration, **optional** (GFPGAN v1.4) | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | Apache-2.0 **plus NVIDIA Source Code License for the StyleGAN2 parts: non-commercial (research/evaluation) use only.** GFPGAN is also built on NVIDIA's pretrained StyleGAN2 weights. |
| `face-detect` | face detection (YuNet 2023mar) | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |

Evaluated in stage 1 but not used: NAFNet SIDD (MIT; shifts colours on camera RAWs).
CodeFormer is not included (S-Lab license, non-commercial).

Architecture code adapted from these repositories lives in
`backend/src/pixforge/models/*_arch.py`; each file header names its origin, license and the
changes made. HAT and SCUNet contain parts derived from SwinIR (Apache-2.0) and timm
(Apache-2.0); RestoreFormer++ from taming-transformers (MIT).

## Python runtime dependencies (shipped)

| Package | Purpose | License |
|---|---|---|
| numpy | arrays | BSD-3-Clause |
| Pillow | image I/O, colour management (bundles LittleCMS: MIT) | MIT-CMU (HPND) |
| opencv-python-headless | resampling, filters, YuNet runtime | Apache-2.0 (OpenCV), MIT (wheel packaging) |
| onnxruntime-directml | inference on DirectML / CPU | MIT |
| truststore | use the Windows certificate store for downloads | MIT |
| pi-heif | HEIC/HEIF decoding (decode-only build, no x265) | BSD-3-Clause; bundles **libheif** and **libde265** under **LGPL-3.0** as separate, replaceable DLLs |
| tifffile | 16-bit TIFF writing | BSD-3-Clause |
| imagecodecs | LZW/ZIP compression for TIFF | BSD-3-Clause (bundled codec libraries under permissive licenses) |
| rawpy (optional `raw` extra) | camera RAW decoding | MIT; bundles **LibRaw** under **LGPL-2.1 or CDDL-1.0** as a separate DLL |

| fastapi, starlette, pydantic | local HTTP API for the app | MIT / BSD-3-Clause / MIT |
| uvicorn, websockets | HTTP/WebSocket server | BSD-3-Clause |

## Desktop app (shipped)

| Package | Purpose | License |
|---|---|---|
| Electron (includes Chromium, Node.js) | desktop shell | MIT (Chromium: BSD-3-Clause and others, see `LICENSES.chromium.html` in the installer) |
| React, React DOM | user interface | MIT |
| zustand | UI state | MIT |
| i18next, react-i18next | German / English texts | MIT |
| lucide-react | icons | ISC |
| Inter (@fontsource-variable/inter) | typeface | SIL Open Font License 1.1 |

Build tools for the app (not shipped): electron-vite, Vite, Tailwind CSS, TypeScript, ESLint,
Prettier, Playwright (MIT / Apache-2.0).

LGPL components are dynamically linked and shipped as separate files; they can be replaced by
the user, and their license texts are included in the installer.

## Developer-only dependencies (not shipped)

| Package | Purpose | License |
|---|---|---|
| torch (CPU) | loading checkpoints, ONNX export | BSD-3-Clause |
| onnx, onnxscript | ONNX export / checks | Apache-2.0 / MIT |
| gdown | downloading upstream weights from Google Drive | MIT |
| pytest, ruff | tests, lint | MIT |

## Test images

Not part of the repository. Developer tests used public-domain photographs from Wikimedia
Commons (Abraham Lincoln portrait, Solvay conference 1927) and private photos of the author,
which stay local.
