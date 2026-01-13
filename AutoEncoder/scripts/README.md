# AutoEncoder

`C:\WorkSpace\Toyota\mag\DataAug\output` に存在する画像を学習するAutoEncoderモデルです。

## 概要

このAutoEncoderは、画像を低次元の潜在表現にエンコードし、元の画像に再構成するモデルです。
- **エンコーダー**: 画像 → 潜在ベクトル（128次元）
- **デコーダー**: 潜在ベクトル → 画像

## ファイル構成

- `model.py`: AutoEncoderモデルの定義
- `dataset.py`: 画像データセットローダー
- `train.py`: 学習スクリプト
- `inference.py`: 推論スクリプト

## 使用方法

### 1. 学習

```bash
python mag/AutoEncoder/train.py \
    --data_dir "C:\WorkSpace\Toyota\mag\DataAug\output" \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001 \
    --latent_dim 128 \
    --image_size 64 64 \
    --save_dir ./checkpoints \
    --log_dir ./logs
```

**主なパラメータ:**
- `--data_dir`: 画像データディレクトリのパス
- `--batch_size`: バッチサイズ（デフォルト: 32）
- `--epochs`: エポック数（デフォルト: 50）
- `--lr`: 学習率（デフォルト: 0.001）
- `--latent_dim`: 潜在空間の次元数（デフォルト: 128）
- `--image_size`: 画像サイズ [height width]（デフォルト: 64 64）
- `--save_dir`: モデル保存ディレクトリ（デフォルト: ./checkpoints）
- `--log_dir`: TensorBoardログ保存ディレクトリ（デフォルト: ./logs）

### 2. 推論（画像再構成）

単一画像を再構成:
```bash
python mag/AutoEncoder/inference.py \
    --checkpoint ./checkpoints/best_model.pth \
    --image_path "path/to/image.png" \
    --output_dir ./reconstructions
```
```
python C:\WorkSpace\Toyota\mag\AutoEncoder\scripts\train.py --data_dir "C:\WorkSpace\Toyota\mag\DataAug\output"   
```


ディレクトリ内の複数画像を再構成:
```bash
python mag/AutoEncoder/inference.py \
    --checkpoint ./checkpoints/best_model.pth \
    --image_dir "C:\WorkSpace\Toyota\mag\DataAug\output" \
    --num_samples 10 \
    --output_dir ./reconstructions
```
```
python C:\WorkSpace\Toyota\mag\AutoEncoder\scripts\inference.py --checkpoint C:\WorkSpace\Toyota\mag\AutoEncoder\checkpoints\best_model.pth --image_dir "C:\WorkSpace\Toyota\mag\DataAug\output" --num_samples 10 --output_dir C:\WorkSpace\Toyota\mag\AutoEncoder\output\reconstructions
```

**主なパラメータ:**
- `--checkpoint`: 学習済みモデルのチェックポイントパス（必須）
- `--image_path`: 再構成する単一画像のパス
- `--image_dir`: 再構成する画像が入ったディレクトリ
- `--num_samples`: image_dir指定時、処理する画像数（デフォルト: 10）
- `--output_dir`: 出力ディレクトリ（デフォルト: ./reconstructions）

## モデル構造

- **エンコーダー**: 4層の畳み込み層で画像を圧縮
  - 64×64 → 32×32 → 16×16 → 8×8 → 4×4
  - 最終的に128次元の潜在ベクトルに変換
  
- **デコーダー**: 4層の転置畳み込み層で画像を再構成
  - 4×4 → 8×8 → 16×16 → 32×32 → 64×64

## 出力

### 学習時
- `checkpoints/best_model.pth`: 検証損失が最小のモデル
- `checkpoints/checkpoint_epoch_N.pth`: 10エポックごとのチェックポイント
- `checkpoints/final_model.pth`: 最終エポックのモデル
- `logs/`: TensorBoardログ（`tensorboard --logdir=./logs`で可視化）

### 推論時
- `reconstructions/*_reconstructed.png`: 再構成画像の可視化（元画像と再構成画像の比較）
- `reconstructions/*_latent.npy`: 潜在ベクトル（numpy配列）

## 依存関係

- torch >= 2.9.1
- torchvision >= 0.24.1
- numpy >= 2.4.1 
- pillow >= 12.1.0
- matplotlib >= 3.8.0
- tqdm >= 4.66.0
- tensorboard >= 2.16.0

