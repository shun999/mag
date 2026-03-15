"""
AutoEncoderの推論スクリプト
学習済みモデルを使用して画像を再構成
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from model import create_model


def load_model(checkpoint_path, device='cuda'):
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


def reconstruct_image(model, image_path, image_size=(64, 64), device='cuda'):
    """単一画像を再構成"""
    # 画像を読み込み
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 変換
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 推論
    with torch.no_grad():
        reconstructed, latent = model(img_tensor)
        reconstructed = reconstructed.cpu().squeeze(0)
    
    # PIL画像に変換
    to_pil = T.ToPILImage()
    original_pil = to_pil(img_tensor.cpu().squeeze(0))
    reconstructed_pil = to_pil(reconstructed)
    
    return original_pil, reconstructed_pil, latent.cpu().numpy()


def visualize_reconstruction(original, reconstructed, save_path=None):
    """再構成結果を可視化"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed)
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='AutoEncoder Inference')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='学習済みモデルのチェックポイントパス'
    )
    parser.add_argument(
        '--image_path',
        type=str,
        help='再構成する画像のパス（単一画像）'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        help='再構成する画像が入ったディレクトリ（複数画像）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./reconstructions',
        help='出力ディレクトリ'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='image_dir指定時、処理する画像数'
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
    
    # モデル読み込み
    print(f"Loading model from: {args.checkpoint}")
    model, model_args = load_model(args.checkpoint, device=device)
    img_size = model_args.get('image_size', [64, 64])
    image_size = tuple(img_size) if isinstance(img_size, (list, tuple)) else (64, 64)
    print(f"Model loaded. Image size: {image_size}")
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 画像パスを収集
    image_paths = []
    if args.image_path:
        image_paths.append(Path(args.image_path))
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            image_paths.extend(list(image_dir.rglob(f'*{ext}')))
            image_paths.extend(list(image_dir.rglob(f'*{ext.upper()}')))
        image_paths = sorted(image_paths)[:args.num_samples]
    else:
        print("Error: --image_path or --image_dir must be specified")
        return
    
    print(f"Processing {len(image_paths)} images...")
    
    # 各画像を処理
    for idx, img_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] Processing: {img_path.name}")
        
        try:
            original, reconstructed, latent = reconstruct_image(
                model, str(img_path), image_size, device
            )
            
            # 保存
            output_path = output_dir / f"{img_path.stem}_reconstructed.png"
            visualize_reconstruction(original, reconstructed, save_path=str(output_path))
            
            # 潜在ベクトルも保存（オプション）
            latent_path = output_dir / f"{img_path.stem}_latent.npy"
            np.save(str(latent_path), latent)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

