# 異常値検出テスト

AutoEncoderモデルを使用した異常値検出テストスクリプトです。

## 概要

学習済みAutoEncoderモデルを用いて、異常画像を検出します。

**検出原理:**
- AutoEncoderは正常データで学習されているため、正常画像は低い再構成誤差を示す
- 異常画像は高い再構成誤差を示す
- 正常データの統計（平均 + N×標準偏差）から閾値を決定
- 再構成誤差が閾値を超えた画像を異常と判定

## 使用方法

### 基本的な使用方法

```bash
python mag/AnomalyDetection/scripts/anomaly_detection.py \
    --checkpoint "C:\WorkSpace\Toyota\mag\AutoEncoder\checkpoints\best_model.pth" \
    --anomaly_data_dir "C:\WorkSpace\Toyota\mag\AnomalyDetection\data\Fe2.5"
```
```
python C:\WorkSpace\Toyota\mag\AnomalyDetection\scripts\anomaly_detection.py --checkpoint "C:\WorkSpace\Toyota\mag\AutoEncoder\checkpoints\final_model.pth" --anomaly_data_dir "C:\WorkSpace\Toyota\mag\AnomalyDetection\data\Fe2.5" --normal_data_dir "C:\WorkSpace\Toyota\mag\AutoEncoder\output\reconstructions"
```

### 全パラメータ指定

```bash
python mag/AnomalyDetection/scripts/anomaly_detection.py \
    --checkpoint "C:\WorkSpace\Toyota\mag\AutoEncoder\checkpoints\best_model.pth" \
    --normal_data_dir "C:\WorkSpace\Toyota\mag\DataAug\output" \
    --anomaly_data_dir "C:\WorkSpace\Toyota\mag\AnomalyDetection\data\Fe2.5" \
    --output_dir "C:\WorkSpace\Toyota\mag\AnomalyDetection\output" \
    --sigma_multiplier 3.0 \
    --num_normal_samples 100 \
    --device auto
```

## パラメータ説明

- `--checkpoint` (必須): 学習済みAutoEncoderモデルのチェックポイントパス
- `--normal_data_dir`: 正常データディレクトリ（閾値計算用、デフォルト: `C:\WorkSpace\Toyota\mag\DataAug\output`）
- `--anomaly_data_dir`: 異常データディレクトリ（検出対象、デフォルト: `C:\WorkSpace\Toyota\mag\AnomalyDetection\data\Fe2.5`）
- `--output_dir`: 出力ディレクトリ（デフォルト: `C:\WorkSpace\Toyota\mag\AnomalyDetection\output`）
- `--threshold`: 閾値（指定しない場合は正常データから自動計算）
- `--sigma_multiplier`: 閾値計算時の標準偏差の倍数（デフォルト: 3.0 = 3σ）
- `--num_normal_samples`: 閾値計算に使用する正常データのサンプル数（デフォルト: 100）
- `--device`: デバイス（auto/cuda/cpu、デフォルト: auto）

## 出力

### 1. CSVファイル: `anomaly_detection_results.csv`

検出結果の詳細が含まれます：
- `image_path`: 画像のフルパス
- `image_name`: 画像ファイル名
- `reconstruction_error`: 再構成誤差（MSE）
- `is_anomaly`: 異常判定（True/False）
- `threshold`: 使用された閾値

### 2. 可視化画像: `*_detection.png`

各画像について、元画像と再構成画像の比較、誤差、判定結果を表示した画像が保存されます。

### 3. 誤差分布グラフ: `error_distribution.png`

正常データと異常データの誤差分布を比較したグラフが保存されます。

## 実行例

```bash
# 基本的な実行
python mag/AnomalyDetection/scripts/anomaly_detection.py \
    --checkpoint "C:\WorkSpace\Toyota\mag\AutoEncoder\checkpoints\best_model.pth"

# 閾値を手動指定
python mag/AnomalyDetection/scripts/anomaly_detection.py \
    --checkpoint "C:\WorkSpace\Toyota\mag\AutoEncoder\checkpoints\best_model.pth" \
    --threshold 0.01

# より厳しい閾値（2σ）を使用
python mag/AnomalyDetection/scripts/anomaly_detection.py \
    --checkpoint "C:\WorkSpace\Toyota\mag\AutoEncoder\checkpoints\best_model.pth" \
    --sigma_multiplier 2.0
```

## 注意事項

1. **モデルの学習**: 異常検出を行う前に、AutoEncoderモデルが正常データで学習されている必要があります。

2. **閾値の調整**: 
   - `sigma_multiplier`を小さくすると、より多くの画像が異常と判定されます（感度が高い）
   - `sigma_multiplier`を大きくすると、より少ない画像が異常と判定されます（特異度が高い）

3. **正常データのサンプル数**: 
   - `num_normal_samples`を増やすと、より正確な閾値が計算されますが、処理時間が長くなります。

4. **画像サイズ**: 
   - モデルの学習時に使用した画像サイズと一致する必要があります（デフォルト: 64×64）
