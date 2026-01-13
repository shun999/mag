"""
AutoEncoderモデル定義
画像を低次元の潜在表現にエンコードし、元の画像にデコードする
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """畳み込みエンコーダー"""
    
    def __init__(self, input_channels=3, latent_dim=128):
        super(ConvEncoder, self).__init__()
        
        # エンコーダー: 画像 -> 潜在ベクトル
        self.encoder = nn.Sequential(
            # 第1ブロック: 64x64 -> 32x32
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第2ブロック: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第3ブロック: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第4ブロック: 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # 潜在ベクトルへの変換
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, latent_dim),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        z = self.fc(x)
        return z


class ConvDecoder(nn.Module):
    """畳み込みデコーダー"""
    
    def __init__(self, latent_dim=128, output_channels=3):
        super(ConvDecoder, self).__init__()
        
        # 潜在ベクトル -> 特徴マップ
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        
        # デコーダー: 特徴マップ -> 画像
        self.decoder = nn.Sequential(
            # 第1ブロック: 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第2ブロック: 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第3ブロック: 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第4ブロック: 32x32 -> 64x64
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # 出力を[0, 1]に正規化
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        x = self.decoder(x)
        return x


class AutoEncoder(nn.Module):
    """AutoEncoderモデル（エンコーダー + デコーダー）"""
    
    def __init__(self, input_channels=3, latent_dim=128):
        super(AutoEncoder, self).__init__()
        self.encoder = ConvEncoder(input_channels, latent_dim)
        self.decoder = ConvDecoder(latent_dim, input_channels)
    
    def forward(self, x):
        # エンコード: 画像 -> 潜在ベクトル
        z = self.encoder(x)
        # デコード: 潜在ベクトル -> 画像
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z
    
    def encode(self, x):
        """画像を潜在ベクトルにエンコード"""
        return self.encoder(x)
    
    def decode(self, z):
        """潜在ベクトルから画像をデコード"""
        return self.decoder(z)


def create_model(input_channels=3, latent_dim=128, device='cuda'):
    """
    AutoEncoderモデルを作成
    
    Args:
        input_channels: 入力画像のチャンネル数（RGB=3, グレースケール=1）
        latent_dim: 潜在空間の次元数
        device: 使用するデバイス（'cuda' or 'cpu'）
    
    Returns:
        AutoEncoderモデル
    """
    model = AutoEncoder(input_channels=input_channels, latent_dim=latent_dim)
    model = model.to(device)
    return model

