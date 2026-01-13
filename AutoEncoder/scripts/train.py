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
    parser = argparse.ArgumentParser(description='Train AutoEncoder')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=r'C:\WorkSpace\Toyota\mag\DataAug\output',
        help='画像データディレクトリのパス'
    )
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=50, help='エポック数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学習率')
    parser.add_argument('--latent_dim', type=int, default=128, help='潜在空間の次元数')
    parser.add_argument('--image_size', type=int, nargs=2, default=[64, 64], help='画像サイズ [height width]')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='モデル保存ディレクトリ')
    parser.add_argument('--log_dir', type=str, default='./logs', help='ログ保存ディレクトリ')
    parser.add_argument('--device', type=str, default='auto', help='デバイス (auto/cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='データローダーのワーカー数')
    
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
    print("-" * 60)
    
    best_val_loss = float('inf')
    
    # 学習ループ
    for epoch in range(1, args.epochs + 1):
        # 学習
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch)
        
        # 検証
        val_loss = validate(model, val_dataloader, criterion, device)
        
        # 学習率スケジューラー更新
        scheduler.step(val_loss)
        
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
            best_model_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }, best_model_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        
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
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {save_dir}")
    print(f"Logs saved to: {log_dir}")
    
    writer.close()


if __name__ == '__main__':
    main()

