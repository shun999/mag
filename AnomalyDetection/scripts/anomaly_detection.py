"""
AutoEncoderを使用した異常値検出スクリプト

改善点:
  - SSIM異常スコア（ピクセル単位の構造的類似度）
  - Mahalanobis距離（潜在空間での統計的距離）
  - MSE + SSIM + Mahalanobis のアンサンブルスコア
  - AUC-ROC / AUC-PR / F1 評価指標
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
import torch.nn.functional as F

# AutoEncoderモジュールをインポート（パスを追加）
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'AIbuild' / 'scripts'))
from model import create_model, SkipAutoEncoder


# ============================================================
# SSIM 計算（ピクセル単位）
# ============================================================
def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma, device)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    k2d = k2d.expand(channels, 1, size, size).contiguous()
    return k2d


def compute_ssim_map(
    x: torch.Tensor, y: torch.Tensor,
    window_size: int = 11, sigma: float = 1.5,
    C1: float = 0.01 ** 2, C2: float = 0.03 ** 2,
) -> tuple:
    """ピクセル単位のSSIMマップとスカラーSSIMを返す"""
    channels = x.size(1)
    kernel = _gaussian_kernel_2d(window_size, sigma, channels, x.device)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den  # [-1, 1]

    return ssim_map, ssim_map.mean().item()


# ============================================================
# Mahalanobis距離
# ============================================================
class MahalanobisCalculator:
    """正常データの潜在ベクトルから統計量を計算し、Mahalanobis距離を算出"""

    def __init__(self):
        self.mean = None
        self.inv_cov = None

    def fit(self, latent_vectors: np.ndarray):
        """正常データの潜在ベクトルから平均・逆共分散行列を算出"""
        self.mean = np.mean(latent_vectors, axis=0)
        cov = np.cov(latent_vectors, rowvar=False)
        # 正則化（特異行列対策）
        cov += np.eye(cov.shape[0]) * 1e-6
        self.inv_cov = np.linalg.inv(cov)

    def distance(self, z: np.ndarray) -> float:
        """1サンプルのMahalanobis距離を返す"""
        diff = z - self.mean
        return float(np.sqrt(diff @ self.inv_cov @ diff))


# ============================================================
# モデル読み込み
# ============================================================
def load_model(checkpoint_path, device='cpu'):
    """学習済みモデルを読み込む"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})

    use_skip = args.get('use_skip', False)

    model = create_model(
        input_channels=args.get('input_channels', 3),
        latent_dim=args.get('latent_dim', 128),
        device=device,
        use_skip=use_skip,
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, args


# ============================================================
# 再構成誤差計算（MSE + SSIM + 潜在ベクトル）
# ============================================================
def calculate_reconstruction_error(model, image_path, image_size=(64, 64), device='cpu'):
    """
    画像の再構成誤差を計算（MSE, SSIM, 潜在ベクトル）

    Returns:
        dict: {mse, ssim, latent_vector} or None
        original_img: 元画像（PIL）
        reconstructed_img: 再構成画像（PIL）
    """
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None

    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed, latent = model(img_tensor)

    # MSE
    mse_error = F.mse_loss(reconstructed, img_tensor, reduction='mean').item()

    # SSIM
    _, ssim_val = compute_ssim_map(reconstructed, img_tensor)

    # 潜在ベクトル
    latent_np = latent.cpu().squeeze(0).numpy()

    to_pil = T.ToPILImage()
    original_pil = to_pil(img_tensor.cpu().squeeze(0))
    reconstructed_pil = to_pil(reconstructed.cpu().squeeze(0))

    result = {
        'mse': mse_error,
        'ssim': ssim_val,
        'latent_vector': latent_np,
    }
    return result, original_pil, reconstructed_pil


# ============================================================
# Grad-CAM
# ============================================================
def _get_gradcam_target_layer(model):
    """モデル構造に応じてGrad-CAMのターゲット層を自動選択"""
    if isinstance(model, SkipAutoEncoder):
        return model.encoder.enc4[0]  # 最終Conv2d層
    else:
        return model.encoder.encoder[-3]  # 従来モデルの最終Conv2d層


def calculate_gradcam_heatmap(model, input_tensor, target_layer=None):
    """再構成誤差を目的関数にしたGrad-CAMヒートマップを計算"""
    if target_layer is None:
        target_layer = _get_gradcam_target_layer(model)

    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def full_backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(full_backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        reconstructed, _ = model(input_tensor)
        loss = F.mse_loss(reconstructed, input_tensor, reduction='mean')
        loss.backward()

        if not activations or not gradients:
            return None

        act = activations[0]
        grad = gradients[0]

        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        return cam[0, 0].detach().cpu().numpy()
    finally:
        f_handle.remove()
        b_handle.remove()


def create_gradcam_overlay(original_pil, heatmap, alpha=0.45):
    """Grad-CAMヒートマップを元画像に重ね合わせる"""
    if heatmap is None:
        return None

    original = np.array(original_pil).astype(np.float32) / 255.0
    heatmap_img = plt.cm.jet(heatmap)[..., :3]
    overlay = np.clip((1 - alpha) * original + alpha * heatmap_img, 0, 1)
    return (overlay * 255).astype(np.uint8)


# ============================================================
# 閾値計算 + Mahalanobis学習
# ============================================================
def calculate_threshold_from_normal_data(
    model, normal_data_dir, image_size=(64, 64), device='cpu',
    num_samples=100, sigma_multiplier=3.0
):
    """
    正常データから閾値を計算し、Mahalanobis計算器を学習

    Returns:
        thresholds: dict {mse_threshold, ssim_threshold, ensemble_threshold}
        stats: dict {mse_mean, mse_std, ssim_mean, ssim_std}
        normal_errors: dict {mse: [...], ssim: [...], ensemble: [...]}
        mahalanobis_calc: MahalanobisCalculator (fitされた状態)
    """
    normal_dir = Path(normal_data_dir)

    image_paths = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        image_paths.extend(list(normal_dir.rglob(f'*{ext}')))
        image_paths.extend(list(normal_dir.rglob(f'*{ext.upper()}')))

    image_paths = sorted(image_paths)[:num_samples]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {normal_data_dir}")

    print(f"Calculating threshold from {len(image_paths)} normal images...")

    mse_errors = []
    ssim_scores = []
    latent_vectors = []

    for img_path in tqdm(image_paths, desc="Processing normal data"):
        result, _, _ = calculate_reconstruction_error(
            model, str(img_path), image_size, device
        )
        if result is not None:
            mse_errors.append(result['mse'])
            ssim_scores.append(result['ssim'])
            latent_vectors.append(result['latent_vector'])

    mse_errors = np.array(mse_errors)
    ssim_scores = np.array(ssim_scores)
    latent_vectors = np.array(latent_vectors)

    # MSE閾値
    mse_mean, mse_std = np.mean(mse_errors), np.std(mse_errors)
    mse_threshold = mse_mean + sigma_multiplier * mse_std

    # SSIM閾値（SSIMは高い=正常なので、低い方向に閾値）
    ssim_mean, ssim_std = np.mean(ssim_scores), np.std(ssim_scores)
    ssim_threshold = ssim_mean - sigma_multiplier * ssim_std

    # Mahalanobis
    mahal_calc = MahalanobisCalculator()
    mahal_calc.fit(latent_vectors)

    mahal_distances = np.array([mahal_calc.distance(z) for z in latent_vectors])
    mahal_mean, mahal_std = np.mean(mahal_distances), np.std(mahal_distances)
    mahal_threshold = mahal_mean + sigma_multiplier * mahal_std

    # アンサンブルスコア: 正規化して統合
    norm_mse = (mse_errors - mse_mean) / (mse_std + 1e-8)
    norm_ssim = -(ssim_scores - ssim_mean) / (ssim_std + 1e-8)  # 反転
    norm_mahal = (mahal_distances - mahal_mean) / (mahal_std + 1e-8)
    ensemble_scores = (norm_mse + norm_ssim + norm_mahal) / 3.0
    ensemble_mean, ensemble_std = np.mean(ensemble_scores), np.std(ensemble_scores)
    ensemble_threshold = ensemble_mean + sigma_multiplier * ensemble_std

    print(f"Normal data statistics:")
    print(f"  MSE    - mean: {mse_mean:.6f}, std: {mse_std:.6f}, threshold: {mse_threshold:.6f}")
    print(f"  SSIM   - mean: {ssim_mean:.6f}, std: {ssim_std:.6f}, threshold: {ssim_threshold:.6f}")
    print(f"  Mahal  - mean: {mahal_mean:.2f}, std: {mahal_std:.2f}, threshold: {mahal_threshold:.2f}")
    print(f"  Ensemble threshold: {ensemble_threshold:.4f}")

    thresholds = {
        'mse': mse_threshold,
        'ssim': ssim_threshold,
        'mahalanobis': mahal_threshold,
        'ensemble': ensemble_threshold,
    }
    stats = {
        'mse_mean': mse_mean, 'mse_std': mse_std,
        'ssim_mean': ssim_mean, 'ssim_std': ssim_std,
        'mahal_mean': mahal_mean, 'mahal_std': mahal_std,
        'ensemble_mean': ensemble_mean, 'ensemble_std': ensemble_std,
    }
    normal_errors = {
        'mse': mse_errors,
        'ssim': ssim_scores,
        'mahalanobis': mahal_distances,
        'ensemble': ensemble_scores,
    }
    return thresholds, stats, normal_errors, mahal_calc


# ============================================================
# 異常検出
# ============================================================
def detect_anomalies(
    model, anomaly_data_dir, thresholds, stats, mahal_calc,
    image_size=(64, 64), device='cpu', output_dir=None, use_gradcam=True
):
    """異常データを検出（MSE + SSIM + Mahalanobis アンサンブル）"""
    anomaly_dir = Path(anomaly_data_dir)

    image_paths = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        image_paths.extend(list(anomaly_dir.rglob(f'*{ext}')))
        image_paths.extend(list(anomaly_dir.rglob(f'*{ext.upper()}')))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {anomaly_data_dir}")

    print(f"\nDetecting anomalies in {len(image_paths)} images...")

    results = []
    output_path = Path(output_dir) if output_dir else None

    for img_path in tqdm(image_paths, desc="Processing anomaly data"):
        result, original, reconstructed = calculate_reconstruction_error(
            model, str(img_path), image_size, device
        )

        if result is not None:
            mse_error = result['mse']
            ssim_val = result['ssim']
            mahal_dist = mahal_calc.distance(result['latent_vector'])

            # 正規化してアンサンブル
            norm_mse = (mse_error - stats['mse_mean']) / (stats['mse_std'] + 1e-8)
            norm_ssim = -(ssim_val - stats['ssim_mean']) / (stats['ssim_std'] + 1e-8)
            norm_mahal = (mahal_dist - stats['mahal_mean']) / (stats['mahal_std'] + 1e-8)
            ensemble_score = (norm_mse + norm_ssim + norm_mahal) / 3.0

            is_anomaly_mse = mse_error > thresholds['mse']
            is_anomaly_ssim = ssim_val < thresholds['ssim']
            is_anomaly_mahal = mahal_dist > thresholds['mahalanobis']
            is_anomaly_ensemble = ensemble_score > thresholds['ensemble']

            results.append({
                'image_path': str(img_path),
                'image_name': img_path.name,
                'mse_error': mse_error,
                'ssim_score': ssim_val,
                'mahalanobis_distance': mahal_dist,
                'ensemble_score': ensemble_score,
                'is_anomaly_mse': is_anomaly_mse,
                'is_anomaly_ssim': is_anomaly_ssim,
                'is_anomaly_mahal': is_anomaly_mahal,
                'is_anomaly_ensemble': is_anomaly_ensemble,
                'threshold_mse': thresholds['mse'],
                'threshold_ssim': thresholds['ssim'],
                'threshold_mahal': thresholds['mahalanobis'],
                'threshold_ensemble': thresholds['ensemble'],
            })

            if output_path:
                gradcam_overlay = None
                if use_gradcam:
                    transform = T.Compose([T.Resize(image_size), T.ToTensor()])
                    input_tensor = transform(original).unsqueeze(0).to(device)
                    heatmap = calculate_gradcam_heatmap(model, input_tensor)
                    gradcam_overlay = create_gradcam_overlay(original, heatmap)

                save_visualization(
                    original, reconstructed, gradcam_overlay,
                    mse_error, ssim_val, ensemble_score,
                    thresholds['ensemble'], is_anomaly_ensemble,
                    output_path / f"{img_path.stem}_detection.png"
                )

    df = pd.DataFrame(results)
    return df


# ============================================================
# 評価指標
# ============================================================
def compute_evaluation_metrics(normal_scores, anomaly_scores, score_name="score"):
    """AUC-ROC, AUC-PR, 最適F1を計算"""
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

    labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    scores = np.concatenate([normal_scores, anomaly_scores])

    auc_roc = roc_auc_score(labels, scores)
    auc_pr = average_precision_score(labels, scores)

    # 最適F1閾値
    precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
    f1_vals = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_vals)
    best_f1 = f1_vals[best_idx]
    best_threshold = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.0

    return {
        f'{score_name}_auc_roc': auc_roc,
        f'{score_name}_auc_pr': auc_pr,
        f'{score_name}_best_f1': best_f1,
        f'{score_name}_best_threshold': best_threshold,
    }


# ============================================================
# 可視化
# ============================================================
def save_visualization(original, reconstructed, gradcam_overlay,
                       mse_error, ssim_val, ensemble_score,
                       threshold, is_anomaly, save_path):
    """検出結果を可視化して保存"""
    has_gradcam = gradcam_overlay is not None
    cols = 3 if has_gradcam else 2
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5))

    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    title = f'Reconstructed\nMSE: {mse_error:.6f}  SSIM: {ssim_val:.4f}'
    if is_anomaly:
        title += f'\nANOMALY (score: {ensemble_score:.3f} > {threshold:.3f})'
    else:
        title += f'\nNormal (score: {ensemble_score:.3f})'
    axes[1].imshow(reconstructed)
    axes[1].set_title(title, fontsize=11, color='red' if is_anomaly else 'green')
    axes[1].axis('off')

    if has_gradcam:
        axes[2].imshow(gradcam_overlay)
        axes[2].set_title('Grad-CAM (Anomaly Localization)', fontsize=12)
        axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_distribution(normal_errors, anomaly_errors, thresholds, save_path=None):
    """誤差分布を可視化（MSE, SSIM, Ensemble）"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics = [
        ('mse', 'MSE Error', False),
        ('ssim', 'SSIM Score', True),  # reversed: low=anomaly
        ('ensemble', 'Ensemble Score', False),
    ]

    for i, (key, label, reversed_) in enumerate(metrics):
        n_vals = normal_errors[key]
        a_vals = anomaly_errors[key] if key in anomaly_errors else []

        # ヒストグラム
        axes[0, i].hist(n_vals, bins=40, alpha=0.7, label='Normal', color='blue', density=True)
        if len(a_vals) > 0:
            axes[0, i].hist(a_vals, bins=40, alpha=0.7, label='Anomaly', color='red', density=True)
        if key in thresholds:
            axes[0, i].axvline(thresholds[key], color='black', linestyle='--', linewidth=2,
                               label=f'Threshold')
        axes[0, i].set_xlabel(label)
        axes[0, i].set_ylabel('Density')
        axes[0, i].set_title(f'{label} Distribution')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

        # ボックスプロット
        box_data = [n_vals]
        box_labels = ['Normal']
        if len(a_vals) > 0:
            box_data.append(a_vals)
            box_labels.append('Anomaly')
        bp = axes[1, i].boxplot(box_data, labels=box_labels, patch_artist=True)
        colors = ['blue', 'red']
        for j, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[j])
            box.set_alpha(0.5)
        if key in thresholds:
            axes[1, i].axhline(thresholds[key], color='black', linestyle='--', linewidth=2)
        axes[1, i].set_ylabel(label)
        axes[1, i].set_title(f'{label} Comparison')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved error distribution plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrices(all_normal_scores, all_anomaly_scores,
                            best_thresholds, save_path=None):
    """4手法の混同行列を2x2サブプロットで可視化"""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    methods = ['mse', 'ssim', 'ensemble', 'mahalanobis']
    display_names = {'mse': 'MSE', 'ssim': 'SSIM',
                     'ensemble': 'Ensemble', 'mahalanobis': 'Mahalanobis'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for idx, method in enumerate(methods):
        ax = axes[idx // 2][idx % 2]
        normal_scores = all_normal_scores[method]
        anomaly_scores = all_anomaly_scores[method]
        threshold = best_thresholds[method]

        labels = np.concatenate([np.zeros(len(normal_scores)),
                                 np.ones(len(anomaly_scores))])
        scores = np.concatenate([normal_scores, anomaly_scores])
        predictions = (scores >= threshold).astype(int)

        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Anomaly'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{display_names[method]}  (threshold={threshold:.4f})',
                     fontsize=12)

    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrices plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curves(all_normal_scores, all_anomaly_scores, save_path=None):
    """4手法のROC曲線を1つのプロットに重ねて描画"""
    from sklearn.metrics import roc_curve, auc

    methods = ['mse', 'ssim', 'ensemble', 'mahalanobis']
    colors = {'mse': 'blue', 'ssim': 'orange',
              'ensemble': 'green', 'mahalanobis': 'red'}
    display_names = {'mse': 'MSE', 'ssim': 'SSIM',
                     'ensemble': 'Ensemble', 'mahalanobis': 'Mahalanobis'}

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for method in methods:
        normal_scores = all_normal_scores[method]
        anomaly_scores = all_anomaly_scores[method]

        labels = np.concatenate([np.zeros(len(normal_scores)),
                                 np.ones(len(anomaly_scores))])
        scores = np.concatenate([normal_scores, anomaly_scores])

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[method], lw=2,
                label=f'{display_names[method]} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curves plot to: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection using AutoEncoder')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='学習済みAutoEncoderモデルのチェックポイントパス')
    parser.add_argument('--normal_data_dir', type=str, default='DataAug/output',
                        help='正常データディレクトリ（閾値計算用）')
    parser.add_argument('--anomaly_data_dir', type=str, default='AnomalyDetection/data/Fe2.5',
                        help='異常データディレクトリ（検出対象）')
    parser.add_argument('--output_dir', type=str, default='AnomalyDetection/output',
                        help='出力ディレクトリ')
    parser.add_argument('--threshold', type=float, default=None,
                        help='MSE閾値（Noneの場合は正常データから自動計算）')
    parser.add_argument('--sigma_multiplier', type=float, default=3.0,
                        help='閾値計算時の標準偏差の倍数（デフォルト: 3.0）')
    parser.add_argument('--num_normal_samples', type=int, default=100,
                        help='閾値計算に使用する正常データのサンプル数')
    parser.add_argument('--device', type=str, default='auto',
                        help='デバイス (auto/cuda/cpu)')
    parser.add_argument('--disable_gradcam', action='store_true',
                        help='Grad-CAM可視化を無効化する')

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
    print(f"Model type: {type(model).__name__}")

    # 閾値計算 + Mahalanobis学習
    print("\n" + "=" * 60)
    print("Calculating thresholds from normal data...")
    print("=" * 60)
    thresholds, stats, normal_errors, mahal_calc = calculate_threshold_from_normal_data(
        model, args.normal_data_dir, image_size, device,
        args.num_normal_samples, args.sigma_multiplier
    )

    # ユーザー指定のMSE閾値があればオーバーライド
    if args.threshold is not None:
        thresholds['mse'] = args.threshold
        print(f"\nOverriding MSE threshold with user value: {args.threshold:.6f}")

    # 異常検出
    print("\n" + "=" * 60)
    print("Detecting anomalies...")
    print("=" * 60)
    results_df = detect_anomalies(
        model, args.anomaly_data_dir, thresholds, stats, mahal_calc,
        image_size, device, output_dir,
        use_gradcam=not args.disable_gradcam
    )

    # 結果を保存
    csv_path = output_dir / 'anomaly_detection_results.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {csv_path}")

    # 統計情報
    total = len(results_df)
    print("\n" + "=" * 60)
    print("Detection Results Summary")
    print("=" * 60)
    print(f"Total images: {total}")
    for method in ['mse', 'ssim', 'mahal', 'ensemble']:
        col = f'is_anomaly_{method}'
        if col in results_df.columns:
            n_anomaly = results_df[col].sum()
            print(f"  {method:>10}: {n_anomaly}/{total} anomalies ({n_anomaly/total*100:.1f}%)")

    # 評価指標（sklearnが利用可能な場合）
    eval_results = {}
    try:
        print("\n" + "=" * 60)
        print("Evaluation Metrics (all anomaly data treated as positive)")
        print("=" * 60)

        anomaly_errors = {
            'mse': results_df['mse_error'].values,
            'ssim': -results_df['ssim_score'].values,  # 反転してhigh=anomaly
            'ensemble': results_df['ensemble_score'].values,
            'mahalanobis': results_df['mahalanobis_distance'].values,
        }

        eval_results = {}
        aligned_normal = {}
        aligned_anomaly = {}
        for score_name, normal_key in [
            ('mse', 'mse'),
            ('ssim', 'ssim'),
            ('ensemble', 'ensemble'),
            ('mahalanobis', 'mahalanobis'),
        ]:
            normal_vals = normal_errors[normal_key]
            anomaly_vals = anomaly_errors[score_name]
            # SSIMは反転済みなので正常データも反転
            if score_name == 'ssim':
                normal_vals = -normal_vals

            aligned_normal[score_name] = normal_vals
            aligned_anomaly[score_name] = anomaly_vals

            metrics = compute_evaluation_metrics(normal_vals, anomaly_vals, score_name)
            eval_results.update(metrics)
            print(f"  {score_name:>10}: AUC-ROC={metrics[f'{score_name}_auc_roc']:.4f}, "
                  f"AUC-PR={metrics[f'{score_name}_auc_pr']:.4f}, "
                  f"Best-F1={metrics[f'{score_name}_best_f1']:.4f}")

        # 混同行列プロット
        best_thresholds = {
            name: eval_results[f'{name}_best_threshold']
            for name in ['mse', 'ssim', 'ensemble', 'mahalanobis']
        }
        cm_path = output_dir / 'confusion_matrices.png'
        plot_confusion_matrices(aligned_normal, aligned_anomaly,
                                best_thresholds, save_path=str(cm_path))

        # ROC曲線プロット
        roc_path = output_dir / 'roc_curves.png'
        plot_roc_curves(aligned_normal, aligned_anomaly, save_path=str(roc_path))

        # 評価指標をCSVに保存
        eval_df = pd.DataFrame([eval_results])
        eval_path = output_dir / 'evaluation_metrics.csv'
        eval_df.to_csv(eval_path, index=False)
        print(f"\nEvaluation metrics saved to: {eval_path}")

    except ImportError:
        print("\nscikit-learn not installed. Skipping evaluation metrics.")
        print("Install with: pip install scikit-learn")
        anomaly_errors = {
            'mse': results_df['mse_error'].values,
            'ssim': results_df['ssim_score'].values,
            'ensemble': results_df['ensemble_score'].values,
        }

    # 検出統計量をnpzファイルに保存（API用）
    stats_data = {
        'mse_mean': np.float64(stats['mse_mean']),
        'mse_std': np.float64(stats['mse_std']),
        'ssim_mean': np.float64(stats['ssim_mean']),
        'ssim_std': np.float64(stats['ssim_std']),
        'mahal_mean': np.float64(stats['mahal_mean']),
        'mahal_std': np.float64(stats['mahal_std']),
        'ensemble_mean': np.float64(stats['ensemble_mean']),
        'ensemble_std': np.float64(stats['ensemble_std']),
        'mse_threshold': np.float64(thresholds['mse']),
        'ssim_threshold': np.float64(thresholds['ssim']),
        'mahal_threshold': np.float64(thresholds['mahalanobis']),
        'ensemble_threshold': np.float64(thresholds['ensemble']),
        'latent_mean': mahal_calc.mean,
        'inv_cov': mahal_calc.inv_cov,
    }
    if eval_results.get('ensemble_best_threshold') is not None:
        stats_data['ensemble_best_threshold'] = np.float64(
            eval_results['ensemble_best_threshold']
        )
    stats_path = output_dir / 'detection_stats.npz'
    np.savez(stats_path, **stats_data)
    print(f"Detection stats saved to: {stats_path}")

    # 誤差分布を可視化
    plot_anomaly_errors = {
        'mse': results_df['mse_error'].values,
        'ssim': results_df['ssim_score'].values,
        'ensemble': results_df['ensemble_score'].values,
    }
    plot_path = output_dir / 'error_distribution.png'
    plot_error_distribution(normal_errors, plot_anomaly_errors, thresholds, save_path=str(plot_path))

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()
