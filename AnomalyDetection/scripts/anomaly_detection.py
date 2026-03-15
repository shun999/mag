"""
AutoEncoderを使用した異常値検出テストスクリプト
学習済みAutoEncoderモデルを用いて、異常画像の検出を行う
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

# AutoEncoderモジュールをインポート（パスを追加）
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'AutoEncoder' / 'scripts'))
from model import create_model


def load_model(checkpoint_path, device='cpu'):
    """学習済みモデルを読み込む"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    
    # モデル作成
    model = create_model(
        input_channels=args.get('input_channels', 3),
        latent_dim=args.get('latent_dim', 128),
        device=device
    )
    
    # 重みを読み込み
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, args


def calculate_reconstruction_error(model, image_path, image_size=(64, 64), device='cpu'):
    """
    画像の再構成誤差を計算
    
    Args:
        model: AutoEncoderモデル
        image_path: 画像パス
        image_size: 画像サイズ (height, width)
        device: デバイス
    
    Returns:
        mse_error: 平均二乗誤差
        original_img: 元画像（PIL）
        reconstructed_img: 再構成画像（PIL）
    """
    # 画像を読み込み
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None
    
    # 変換
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 推論
    with torch.no_grad():
        reconstructed, _ = model(img_tensor)
    
    # 再構成誤差を計算（MSE）
    mse = nn.MSELoss(reduction='mean')
    mse_error = mse(reconstructed, img_tensor).item()
    
    # PIL画像に変換
    to_pil = T.ToPILImage()
    original_pil = to_pil(img_tensor.cpu().squeeze(0))
    reconstructed_pil = to_pil(reconstructed.cpu().squeeze(0))
    
    return mse_error, original_pil, reconstructed_pil


def calculate_threshold_from_normal_data(
    model, normal_data_dir, image_size=(64, 64), device='cpu', 
    num_samples=100, sigma_multiplier=3.0
):
    """
    正常データから閾値を計算
    
    Args:
        model: AutoEncoderモデル
        normal_data_dir: 正常データディレクトリ
        image_size: 画像サイズ
        device: デバイス
        num_samples: サンプル数
        sigma_multiplier: 標準偏差の倍数（デフォルト: 3σ）
    
    Returns:
        threshold: 閾値
        mean_error: 平均誤差
        std_error: 標準偏差
        errors: 誤差のリスト
    """
    normal_dir = Path(normal_data_dir)
    
    # 画像ファイルを収集
    image_paths = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        image_paths.extend(list(normal_dir.rglob(f'*{ext}')))
        image_paths.extend(list(normal_dir.rglob(f'*{ext.upper()}')))
    
    image_paths = sorted(image_paths)[:num_samples]
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {normal_data_dir}")
    
    print(f"Calculating threshold from {len(image_paths)} normal images...")
    
    errors = []
    for img_path in tqdm(image_paths, desc="Processing normal data"):
        error, _, _ = calculate_reconstruction_error(
            model, str(img_path), image_size, device
        )
        if error is not None:
            errors.append(error)
    
    errors = np.array(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    threshold = mean_error + sigma_multiplier * std_error
    
    print(f"Normal data statistics:")
    print(f"  Mean error: {mean_error:.6f}")
    print(f"  Std error: {std_error:.6f}")
    print(f"  Threshold (mean + {sigma_multiplier}σ): {threshold:.6f}")
    
    return threshold, mean_error, std_error, errors


def detect_anomalies(
    model, anomaly_data_dir, threshold, image_size=(64, 64), 
    device='cpu', output_dir=None
):
    """
    異常データを検出
    
    Args:
        model: AutoEncoderモデル
        anomaly_data_dir: 異常データディレクトリ
        threshold: 閾値
        image_size: 画像サイズ
        device: デバイス
        output_dir: 出力ディレクトリ
    
    Returns:
        results: 結果のDataFrame
    """
    anomaly_dir = Path(anomaly_data_dir)
    
    # 画像ファイルを収集
    image_paths = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        image_paths.extend(list(anomaly_dir.rglob(f'*{ext}')))
        image_paths.extend(list(anomaly_dir.rglob(f'*{ext.upper()}')))
    
    image_paths = sorted(image_paths)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {anomaly_data_dir}")
    
    print(f"\nDetecting anomalies in {len(image_paths)} images...")
    print(f"Threshold: {threshold:.6f}")
    
    results = []
    output_path = Path(output_dir) if output_dir else None
    
    for img_path in tqdm(image_paths, desc="Processing anomaly data"):
        error, original, reconstructed = calculate_reconstruction_error(
            model, str(img_path), image_size, device
        )
        
        if error is not None:
            is_anomaly = error > threshold
            results.append({
                'image_path': str(img_path),
                'image_name': img_path.name,
                'reconstruction_error': error,
                'is_anomaly': is_anomaly,
                'threshold': threshold,
            })
            
            # 可視化を保存
            if output_path:
                save_visualization(
                    original, reconstructed, error, threshold, is_anomaly,
                    output_path / f"{img_path.stem}_detection.png"
                )
    
    df = pd.DataFrame(results)
    return df


def save_visualization(original, reconstructed, error, threshold, is_anomaly, save_path):
    """検出結果を可視化して保存"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 元画像
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # 再構成画像
    axes[1].imshow(reconstructed)
    title = f'Reconstructed Image\nError: {error:.6f}'
    if is_anomaly:
        title += f'\n⚠ ANOMALY DETECTED (>{threshold:.6f})'
    else:
        title += f'\n✓ Normal (≤{threshold:.6f})'
    axes[1].set_title(title, fontsize=12, color='red' if is_anomaly else 'green')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_distribution(normal_errors, anomaly_errors, threshold, save_path=None):
    """誤差分布を可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ヒストグラム
    axes[0].hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    axes[0].hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    axes[0].axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.6f})')
    axes[0].set_xlabel('Reconstruction Error (MSE)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ボックスプロット
    box_data = [normal_errors, anomaly_errors]
    bp = axes[1].boxplot(box_data, labels=['Normal', 'Anomaly'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    axes[1].axhline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.6f})')
    axes[1].set_ylabel('Reconstruction Error (MSE)')
    axes[1].set_title('Error Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error distribution plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection using AutoEncoder')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='学習済みAutoEncoderモデルのチェックポイントパス'
    )
    parser.add_argument(
        '--normal_data_dir',
        type=str,
        default=r'C:\WorkSpace\Toyota\mag\DataAug\output',
        help='正常データディレクトリ（閾値計算用）'
    )
    parser.add_argument(
        '--anomaly_data_dir',
        type=str,
        default=r'C:\WorkSpace\Toyota\mag\AnomalyDetection\data\Fe2.5',
        help='異常データディレクトリ（検出対象）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r'C:\WorkSpace\Toyota\mag\AnomalyDetection\output',
        help='出力ディレクトリ'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='閾値（Noneの場合は正常データから自動計算）'
    )
    parser.add_argument(
        '--sigma_multiplier',
        type=float,
        default=3.0,
        help='閾値計算時の標準偏差の倍数（デフォルト: 3.0）'
    )
    parser.add_argument(
        '--num_normal_samples',
        type=int,
        default=100,
        help='閾値計算に使用する正常データのサンプル数'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='デバイス (auto/cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデル読み込み
    print(f"\nLoading model from: {args.checkpoint}")
    model, model_args = load_model(args.checkpoint, device=device)
    img_size = model_args.get('image_size', [64, 64])
    image_size = tuple(img_size) if isinstance(img_size, (list, tuple)) else (64, 64)
    print(f"Model loaded. Image size: {image_size}")
    
    # 閾値計算
    if args.threshold is None:
        print("\n" + "="*60)
        print("Calculating threshold from normal data...")
        print("="*60)
        threshold, mean_error, std_error, normal_errors = calculate_threshold_from_normal_data(
            model, args.normal_data_dir, image_size, device,
            args.num_normal_samples, args.sigma_multiplier
        )
    else:
        threshold = args.threshold
        print(f"\nUsing provided threshold: {threshold:.6f}")
        # 正常データの誤差も計算（可視化用）
        _, _, _, normal_errors = calculate_threshold_from_normal_data(
            model, args.normal_data_dir, image_size, device,
            args.num_normal_samples, args.sigma_multiplier
        )
    
    # 異常検出
    print("\n" + "="*60)
    print("Detecting anomalies...")
    print("="*60)
    results_df = detect_anomalies(
        model, args.anomaly_data_dir, threshold, image_size, device, output_dir
    )
    
    # 結果を保存
    csv_path = output_dir / 'anomaly_detection_results.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {csv_path}")
    
    # 統計情報
    total = len(results_df)
    anomalies = results_df['is_anomaly'].sum()
    normal = total - anomalies
    
    print("\n" + "="*60)
    print("Detection Results Summary")
    print("="*60)
    print(f"Total images: {total}")
    print(f"Anomalies detected: {anomalies} ({anomalies/total*100:.2f}%)")
    print(f"Normal: {normal} ({normal/total*100:.2f}%)")
    print(f"Threshold: {threshold:.6f}")
    print(f"Mean error (anomaly data): {results_df['reconstruction_error'].mean():.6f}")
    print(f"Std error (anomaly data): {results_df['reconstruction_error'].std():.6f}")
    print(f"Min error: {results_df['reconstruction_error'].min():.6f}")
    print(f"Max error: {results_df['reconstruction_error'].max():.6f}")
    
    # 異常検出された画像のリスト
    if anomalies > 0:
        anomaly_images = results_df[results_df['is_anomaly']]['image_name'].tolist()
        print(f"\nAnomaly images ({anomalies}):")
        for img in anomaly_images[:10]:  # 最初の10個を表示
            print(f"  - {img}")
        if len(anomaly_images) > 10:
            print(f"  ... and {len(anomaly_images) - 10} more")
    
    # 誤差分布を可視化
    anomaly_errors = results_df['reconstruction_error'].values
    plot_path = output_dir / 'error_distribution.png'
    plot_error_distribution(normal_errors, anomaly_errors, threshold, save_path=str(plot_path))
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
