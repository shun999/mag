"""
AutoEncoderモデル定義
画像を低次元の潜在表現にエンコードし、元の画像にデコードする

アーキテクチャ:
  - SkipAutoEncoder (use_skip=True): U-Net風Skip Connection + SE Block
  - AutoEncoder (use_skip=False): 従来の標準AutoEncoder（後方互換）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SE Block (Squeeze-and-Excitation)
# ============================================================
class SEBlock(nn.Module):
    """チャンネルごとの重要度を学習するAttentionモジュール"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ============================================================
# 従来モデル（後方互換のため残す）
# ============================================================
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
            nn.Sigmoid(),
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
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


# ============================================================
# 新モデル: U-Net風 Skip Connection + SE Block
# ============================================================
class SkipEncoder(nn.Module):
    """SE Block付きエンコーダー。各層の中間特徴マップを返す"""

    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()

        # 第1ブロック: 64x64 -> 32x32
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.se1 = SEBlock(64)

        # 第2ブロック: 32x32 -> 16x16
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.se2 = SEBlock(128)

        # 第3ブロック: 16x16 -> 8x8
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.se3 = SEBlock(256)

        # 第4ブロック: 8x8 -> 4x4
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.se4 = SEBlock(512)

        # 潜在ベクトルへの変換
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, latent_dim),
        )

    def forward(self, x):
        s1 = self.se1(self.enc1(x))   # [B, 64,  32, 32]
        s2 = self.se2(self.enc2(s1))   # [B, 128, 16, 16]
        s3 = self.se3(self.enc3(s2))   # [B, 256,  8,  8]
        s4 = self.se4(self.enc4(s3))   # [B, 512,  4,  4]
        z = self.fc(s4)                # [B, latent_dim]
        return z, (s1, s2, s3)


class SkipDecoder(nn.Module):
    """Skip Connectionを受け取るデコーダー。
    EncoderのS1/S2/S3を対応する層にconcatし、1x1 Convでチャンネルを戻す。
    """

    def __init__(self, latent_dim=128, output_channels=3):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # 4x4 -> 8x8  (+ skip s3: 256ch)  => 256+256=512 -> 256
        self.dec1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.fuse1 = nn.Conv2d(256 + 256, 256, kernel_size=1)

        # 8x8 -> 16x16  (+ skip s2: 128ch)  => 128+128=256 -> 128
        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fuse2 = nn.Conv2d(128 + 128, 128, kernel_size=1)

        # 16x16 -> 32x32  (+ skip s1: 64ch)  => 64+64=128 -> 64
        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fuse3 = nn.Conv2d(64 + 64, 64, kernel_size=1)

        # 32x32 -> 64x64
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, skips):
        s1, s2, s3 = skips

        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)

        # 4x4 -> 8x8 + skip s3
        x = F.relu(self.bn1(self.dec1(x)), inplace=True)
        x = F.relu(self.fuse1(torch.cat([x, s3], dim=1)), inplace=True)

        # 8x8 -> 16x16 + skip s2
        x = F.relu(self.bn2(self.dec2(x)), inplace=True)
        x = F.relu(self.fuse2(torch.cat([x, s2], dim=1)), inplace=True)

        # 16x16 -> 32x32 + skip s1
        x = F.relu(self.bn3(self.dec3(x)), inplace=True)
        x = F.relu(self.fuse3(torch.cat([x, s1], dim=1)), inplace=True)

        # 32x32 -> 64x64
        x = self.dec4(x)
        return x


class SkipAutoEncoder(nn.Module):
    """U-Net風Skip Connection + SE Block付きAutoEncoder"""

    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()
        self.encoder = SkipEncoder(input_channels, latent_dim)
        self.decoder = SkipDecoder(latent_dim, input_channels)

    def forward(self, x):
        z, skips = self.encoder(x)
        x_reconstructed = self.decoder(z, skips)
        return x_reconstructed, z

    def encode(self, x):
        z, _ = self.encoder(x)
        return z

    def decode(self, z):
        # Skip connection無しのデコードはfallback（ONNX export用など）
        # ダミーskipsを生成
        batch = z.size(0)
        s1 = torch.zeros(batch, 64, 32, 32, device=z.device)
        s2 = torch.zeros(batch, 128, 16, 16, device=z.device)
        s3 = torch.zeros(batch, 256, 8, 8, device=z.device)
        return self.decoder(z, (s1, s2, s3))


# ============================================================
# ファクトリ関数
# ============================================================
def create_model(input_channels=3, latent_dim=128, device='cuda', use_skip=True):
    """
    AutoEncoderモデルを作成

    Args:
        input_channels: 入力画像のチャンネル数（RGB=3, グレースケール=1）
        latent_dim: 潜在空間の次元数
        device: 使用するデバイス（'cuda' or 'cpu'）
        use_skip: True=SkipAutoEncoder（推奨）, False=従来のAutoEncoder

    Returns:
        AutoEncoderモデル
    """
    if use_skip:
        model = SkipAutoEncoder(input_channels=input_channels, latent_dim=latent_dim)
    else:
        model = AutoEncoder(input_channels=input_channels, latent_dim=latent_dim)
    model = model.to(device)
    return model
