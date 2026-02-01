"""
AutoEncoderの学習スクリプト
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model import create_model
from dataset import create_dataloader


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """1エポックの学習"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, images in enumerate(pbar):
        images = images.to(device)
        
        # 順伝播
        reconstructed, latent = model(images)
        
        # 損失計算（再構成誤差）
        loss = criterion(reconstructed, images)
        
        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # プログレスバーを更新
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """検証"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            
            reconstructed, latent = model(images)
            loss = criterion(reconstructed, images)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    output_path = Path('/media/dl-box/ADATA SE800/Toyota/mag/AutoEncoder/output')
    parser = argparse.ArgumentParser(description='Train AutoEncoder')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=r'/media/dl-box/ADATA SE800/Toyota/mag/DataAug/output',
        help='画像データディレクトリのパス'
    )
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=3, help='エポック数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学習率')
    parser.add_argument('--latent_dim', type=int, default=128, help='潜在空間の次元数')
    parser.add_argument('--image_size', type=int, nargs=2, default=[64, 64], help='画像サイズ [height width]')
    parser.add_argument('--save_dir', type=str, default= output_path/'checkpoints', help='モデル保存ディレクトリ')
    parser.add_argument('--log_dir', type=str, default=output_path/'logs', help='ログ保存ディレクトリ')
    parser.add_argument('--device', type=str, default='auto', help='デバイス (auto/cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='データローダーのワーカー数')
    parser.add_argument('--early_stopping', action='store_true', help='Early Stoppingを使用する')
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='Early Stoppingの忍耐度（エポック数）')
    parser.add_argument('--plot_dir', type=str, default=output_path/'plots', help='グラフ保存ディレクトリ')
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
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
    
    # 検証用データローダー（シャッフルなし）
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
        device=device
    )
    
    # 損失関数とオプティマイザー
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学習率スケジューラー
    # 一部のPyTorchバージョンではverbose引数が存在しないため指定しない
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Early Stopping: {'Enabled' if args.early_stopping else 'Disabled'}")
    print("-" * 60)
    
    # 学習用の履歴を保存するリスト
    train_losses = []
    val_losses = []
    learning_rates = []
    
    best_val_loss = float('inf')
    
    # Early Stopping用の変数
    early_stop_counter = 0
    
    # 学習ループ
    for epoch in range(1, args.epochs + 1):
        # 学習
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch)
        
        # 検証
        val_loss = validate(model, val_dataloader, criterion, device)
        
        # 学習率スケジューラー更新
        scheduler.step(val_loss)
        
        # 履歴に追加
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # ログ記録
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # ベストモデルを保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0  # Early Stoppingカウンターをリセット
            best_model_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }, best_model_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        else:
            # Early Stopping用カウンター増加
            if args.early_stopping:
                early_stop_counter += 1
                print(f"  ⚠ No improvement. Early stopping counter: {early_stop_counter}/{args.early_stopping_patience}")
                
                # Early Stopping判定
                if early_stop_counter >= args.early_stopping_patience:
                    print(f"\n⛔ Early stopping triggered at epoch {epoch}")
                    break
        
        # 定期的にチェックポイントを保存
        if epoch % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }, checkpoint_path)
        
        print("-" * 60)
    
    # 最終モデルを保存
    final_model_path = save_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'args': vars(args),
    }, final_model_path)
    
    # 学習曲線を描画・保存
    plot_training_curves(train_losses, val_losses, learning_rates, plot_dir)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {save_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"Plots saved to: {plot_dir}")
    
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
    
    # 学習率を右軸に
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', fontsize=12, color='tab:green')
    line3 = ax2.plot(epochs, learning_rates, label='Learning Rate', color='tab:green', 
                     linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:green')
    
    # タイトルと凡例
    plt.title('Training Curves', fontsize=14, fontweight='bold')
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10)
    
    # 保存
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
    
    # 最小値をマーク
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

