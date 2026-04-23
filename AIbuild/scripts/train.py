"""
AutoEncoderの学習スクリプト

改善点:
  - SSIM + MSE 複合損失関数
  - CosineAnnealingWarmRestarts スケジューラー
  - Gradient Clipping
  - Mixed Precision Training (AMP)
  - SkipAutoEncoder / 従来AutoEncoder の切り替え
  - MLflow トラッキング (パラメータ・メトリクス・モデル・アーティファクト)
"""

import argparse
import os
import math
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import create_model
from dataset import create_dataloader


# ============================================================
# SSIM 損失
# ============================================================
def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma, device)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)  # outer product
    k2d = k2d.expand(channels, 1, size, size).contiguous()
    return k2d


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """バッチ平均SSIMを返す (値域 [-1, 1])"""
    # AMP (float16) では分散計算が不安定になるため float32 で実行
    with torch.amp.autocast(device_type='cuda', enabled=False):
        x = x.float()
        y = y.float()

        channels = x.size(1)
        kernel = _gaussian_kernel_2d(window_size, sigma, channels, x.device)

        mu_x = torch_F.conv2d(x, kernel, padding=window_size // 2, groups=channels)
        mu_y = torch_F.conv2d(y, kernel, padding=window_size // 2, groups=channels)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x2 = torch_F.conv2d(x * x, kernel, padding=window_size // 2, groups=channels) - mu_x2
        sigma_y2 = torch_F.conv2d(y * y, kernel, padding=window_size // 2, groups=channels) - mu_y2
        sigma_xy = torch_F.conv2d(x * y, kernel, padding=window_size // 2, groups=channels) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

        ssim_map = num / den
        return ssim_map.mean()


class CombinedLoss(nn.Module):
    """MSE + α × (1 - SSIM) 複合損失"""

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_val = ssim(pred.float(), target.float())
        ssim_loss = 1.0 - ssim_val
        return mse_loss + self.alpha * ssim_loss


# ============================================================
# 学習・検証
# ============================================================
def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None, max_grad_norm=1.0):
    """1エポックの学習（AMP対応）"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, images in enumerate(pbar):
        images = images.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                reconstructed, latent = model(images)
                loss = criterion(reconstructed, images)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed, latent = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device, scaler=None):
    """検証（AMP対応）"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    with torch.no_grad():
        for images in tqdm(dataloader, desc="Validation"):
            images = images.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    reconstructed, latent = model(images)
                    loss = criterion(reconstructed, images)
            else:
                reconstructed, latent = model(images)
                loss = criterion(reconstructed, images)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def _save_checkpoint(model, optimizer, epoch, val_loss, args, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'args': vars(args),
    }, path)


# ============================================================
# main
# ============================================================
def main():
    output_path = Path('AIbuild/output')
    parser = argparse.ArgumentParser(description='Train AutoEncoder')
    parser.add_argument('--data_dir', type=str, default='DataAug/output', help='画像データディレクトリのパス')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=500, help='エポック数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学習率')
    parser.add_argument('--latent_dim', type=int, default=128, help='潜在空間の次元数')
    parser.add_argument('--image_size', type=int, nargs=2, default=[64, 64], help='画像サイズ [height width]')
    parser.add_argument('--save_dir', type=str, default=str(output_path / 'checkpoints'), help='モデル保存ディレクトリ')
    parser.add_argument('--log_dir', type=str, default=str(output_path / 'logs'), help='ログ保存ディレクトリ')
    parser.add_argument('--plot_dir', type=str, default=str(output_path / 'plots'), help='グラフ保存ディレクトリ')
    parser.add_argument('--device', type=str, default='auto', help='デバイス (auto/cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='データローダーのワーカー数')
    parser.add_argument('--early_stopping', action='store_true', help='Early Stoppingを使用する')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='Early Stoppingの忍耐度（エポック数）')
    # 新オプション
    parser.add_argument('--use_skip', action='store_true', default=True,
                        help='SkipAutoEncoder (U-Net風) を使用する（デフォルト: True）')
    parser.add_argument('--no_skip', dest='use_skip', action='store_false',
                        help='従来のAutoEncoderを使用する')
    parser.add_argument('--ssim_alpha', type=float, default=0.5,
                        help='SSIM損失の重み (0=MSEのみ, 1=SSIMのみ)')
    parser.add_argument('--loss', type=str, default='combined', choices=['mse', 'combined'],
                        help='損失関数 (mse / combined=MSE+SSIM)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['plateau', 'cosine'],
                        help='学習率スケジューラー')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Mixed Precision Training (AMP) を使用する')
    parser.add_argument('--no_amp', dest='amp', action='store_false',
                        help='AMPを無効化する')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Gradient Clippingの最大ノルム')
    # MLflow オプション
    parser.add_argument('--experiment_name', type=str, default='AutoEncoder',
                        help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='MLflow run name (省略時は自動生成)')

    args = parser.parse_args()

    # デバイス設定
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # AMPはCUDAでのみ有効
    use_amp = args.amp and device.type == 'cuda'

    print(f"Using device: {device}")
    print(f"AMP: {'Enabled' if use_amp else 'Disabled'}")
    print(f"Model: {'SkipAutoEncoder' if args.use_skip else 'AutoEncoder'}")
    print(f"Loss: {args.loss} (ssim_alpha={args.ssim_alpha})")
    print(f"Scheduler: {args.scheduler}")

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):

        # ディレクトリ作成
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoardライター
        writer = SummaryWriter(log_dir=str(log_dir))

        # データローダー作成
        print("Loading dataset...")
        dataloader = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=tuple(args.image_size),
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda',
        )

        val_dataloader = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=tuple(args.image_size),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == 'cuda',
        )

        # モデル作成
        print("Creating model...")
        model = create_model(
            input_channels=3,
            latent_dim=args.latent_dim,
            device=device,
            use_skip=args.use_skip,
        )
        total_params = sum(p.numel() for p in model.parameters())

        # 損失関数
        if args.loss == 'combined':
            criterion = CombinedLoss(alpha=args.ssim_alpha)
        else:
            criterion = nn.MSELoss()

        # オプティマイザー
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 学習率スケジューラー
        if args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=50, T_mult=2
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

        # AMP Scaler
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        print(f"Model parameters: {total_params:,}")
        print(f"Starting training for {args.epochs} epochs...")
        print(f"Early Stopping: {'Enabled' if args.early_stopping else 'Disabled'}")
        print("-" * 60)

        # ──────────────────────────────────────────────────────
        # MLflow: ハイパーパラメータを記録
        # ──────────────────────────────────────────────────────
        mlflow.log_params({
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "latent_dim": args.latent_dim,
            "image_size": f"{args.image_size[0]}x{args.image_size[1]}",
            "loss": args.loss,
            "ssim_alpha": args.ssim_alpha,
            "scheduler": args.scheduler,
            "amp": use_amp,
            "max_grad_norm": args.max_grad_norm,
            "early_stopping": args.early_stopping,
            "early_stopping_patience": args.early_stopping_patience,
            "use_skip": args.use_skip,
            "model_type": "SkipAutoEncoder" if args.use_skip else "AutoEncoder",
            "total_params": total_params,
            "data_dir": args.data_dir,
        })

        # 学習用の履歴
        train_losses = []
        val_losses = []
        learning_rates = []

        best_val_loss = float('inf')
        best_epoch = 0
        early_stop_counter = 0

        # 学習ループ
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model, dataloader, criterion, optimizer, device, epoch,
                scaler=scaler, max_grad_norm=args.max_grad_norm,
            )
            val_loss = validate(model, val_dataloader, criterion, device, scaler=scaler)

            # スケジューラー更新
            if args.scheduler == 'cosine':
                scheduler.step(epoch)
            else:
                scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]['lr']

            # 履歴に追加
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            learning_rates.append(current_lr)

            # TensorBoard ログ
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)

            # ──────────────────────────────────────────────────
            # MLflow: エポック毎メトリクスを記録
            # ──────────────────────────────────────────────────
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
            }, step=epoch)

            print(f"Epoch {epoch}/{args.epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {current_lr:.6f}")

            # ベストモデルを保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                early_stop_counter = 0
                best_ckpt_path = save_dir / 'best_model.pth'
                _save_checkpoint(model, optimizer, epoch, val_loss, args, best_ckpt_path)
                print(f"  Saved best model (val_loss: {val_loss:.6f})")

                # ──────────────────────────────────────────────
                # MLflow: ベストモデルを上書き記録
                # ──────────────────────────────────────────────
                mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
            else:
                if args.early_stopping:
                    early_stop_counter += 1
                    print(f"  No improvement. Early stopping counter: {early_stop_counter}/{args.early_stopping_patience}")
                    if early_stop_counter >= args.early_stopping_patience:
                        print(f"\nEarly stopping triggered at epoch {epoch}")
                        break

            # 定期的にチェックポイントを保存
            if epoch % 10 == 0:
                _save_checkpoint(model, optimizer, epoch, val_loss, args,
                                 save_dir / f'checkpoint_epoch_{epoch}.pth')

            print("-" * 60)

        # 最終モデルを保存
        _save_checkpoint(model, optimizer, epoch, val_loss, args,
                         save_dir / 'final_model.pth')

        # 学習曲線を描画・保存
        plot_training_curves(train_losses, val_losses, learning_rates, plot_dir)

        # ──────────────────────────────────────────────────────
        # MLflow: サマリーメトリクス・アーティファクトを記録
        # ──────────────────────────────────────────────────────
        mlflow.log_metrics({
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "actual_epochs": epoch,
        })
        mlflow.log_artifact(str(save_dir / 'best_model.pth'), artifact_path="checkpoints")
        mlflow.log_artifacts(str(plot_dir), artifact_path="plots")

        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"Models saved to: {save_dir}")
        print(f"Logs saved to: {log_dir}")
        print(f"Plots saved to: {plot_dir}")
        print(f"MLflow experiment: {args.experiment_name}")

        writer.close()


def plot_training_curves(train_losses, val_losses, learning_rates, save_dir):
    """学習曲線を描画して保存"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 損失の曲線
    fig, ax1 = plt.subplots(figsize=(12, 5))

    epochs = np.arange(1, len(train_losses) + 1)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='tab:blue')
    line1 = ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue', linewidth=2)
    line2 = ax1.plot(epochs, val_losses, label='Validation Loss', color='tab:orange', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', fontsize=12, color='tab:green')
    line3 = ax2.plot(epochs, learning_rates, label='Learning Rate', color='tab:green',
                     linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title('Training Curves', fontsize=14, fontweight='bold')
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)

    loss_plot_path = save_dir / 'training_curves.png'
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {loss_plot_path}")
    plt.close()

    # 損失のみの拡大図
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    min_val_loss_epoch = np.argmin(val_losses) + 1
    min_val_loss = np.min(val_losses)
    ax.annotate(f'Min: {min_val_loss:.4f}',
                xy=(min_val_loss_epoch, min_val_loss),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    loss_detail_path = save_dir / 'loss_detail.png'
    plt.tight_layout()
    plt.savefig(loss_detail_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss detail to: {loss_detail_path}")
    plt.close()

    # 学習率の曲線
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, learning_rates, linewidth=2, color='tab:green', marker='o', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    lr_plot_path = save_dir / 'learning_rate.png'
    plt.tight_layout()
    plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning rate plot to: {lr_plot_path}")
    plt.close()


if __name__ == '__main__':
    main()
