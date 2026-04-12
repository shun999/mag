# mag

画像から異常検知を行うシステムの開発を目指すプロジェクトです。
- "./DataAug"で撮影した画像データのデータ数を増やします。
- "./AIbuild"で、"./DataAug"でデータ拡張された画像データを用いて、AIモデルの学習を行います。
- "./AnomalyDetection"で、構築したAIモデルを実際の画像を用いてテストを行います。
- "./api"で、構築したAIモデルを、FastAPIでAPI化します。


## Environment Setup

Dependencies are managed with `uv` (Python 3.10–3.11 required). PyTorch is installed from the CUDA 11.7 index — do not change the index source.

```bash
uv sync
```

### spec of pc used for AI model development

NVIDIA-SMI 515.43.04    Driver Version: 515.43.04    CUDA Version: 11.7
python version: 3.11.14
Pytorch version: 2.0.1+cu117
GPU: RTX A6000

## Pipeline Overview

This project is an **image anomaly detection system** for manufacturing inspection. The pipeline flows left to right:

```
DataAug → AIbuild → AnomalyDetection → API
```

## Branch Strategy

- `main` — stable / release
- `dev` — active general development
- `devcc` — active development of claude code

## 注意事項
"./DataAug/data"には、画像の元データがあるので、編集しないでください。