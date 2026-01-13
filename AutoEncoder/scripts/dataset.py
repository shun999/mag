"""
画像データセットローダー
outputディレクトリ内の全画像を読み込む
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class ImageDataset(Dataset):
    """画像データセット"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (64, 64),
        transform: Optional[T.Compose] = None,
    ):
        """
        Args:
            data_dir: 画像が格納されているディレクトリパス
            image_size: リサイズ後の画像サイズ (height, width)
            transform: 追加の変換（Noneの場合はデフォルト変換を使用）
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # 画像ファイルのパスを収集（再帰的に）
        self.image_paths = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            self.image_paths.extend(list(self.data_dir.rglob(f'*{ext}')))
            self.image_paths.extend(list(self.data_dir.rglob(f'*{ext.upper()}')))
        
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        # デフォルトの変換
        if transform is None:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.ToTensor(),  # [0, 255] -> [0, 1] に正規化
            ])
        else:
            self.transform = transform
        
        print(f"Found {len(self.image_paths)} images in {data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # 画像を読み込み
            img = Image.open(img_path)
            
            # RGBに変換（グレースケールやRGBAの場合）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 変換を適用
            img_tensor = self.transform(img)
            
            return img_tensor
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # エラー時は黒画像を返す
            return torch.zeros(3, self.image_size[0], self.image_size[1])


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (64, 64),
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    データローダーを作成
    
    Args:
        data_dir: 画像ディレクトリパス
        batch_size: バッチサイズ
        image_size: 画像サイズ (height, width)
        shuffle: シャッフルするかどうか
        num_workers: データローディングのワーカー数
        pin_memory: GPU使用時にTrue推奨
    
    Returns:
        DataLoader
    """
    dataset = ImageDataset(data_dir=data_dir, image_size=image_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return dataloader

