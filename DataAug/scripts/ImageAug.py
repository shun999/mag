"""
フォルダ内の全画像を対象に、
「①幾何学変換＋②輝度変換＋③ノイズ付加」を組み合わせて
各画像から500枚ずつ拡張して保存するスクリプト。

- 入力: input_dir（画像が入ったフォルダ）
- 出力: output_dir / <元ファイル名>/ に 500枚保存
  例: output/normal_001/normal_001_0001.png ... normal_001_0500.png

依存:
  pip install pillow torchvision torch numpy
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


def add_gaussian_noise_tensor(img_t: torch.Tensor, sigma: float) -> torch.Tensor:
    """img_t: [C,H,W] in [0,1]"""
    if sigma <= 0:
        return img_t
    noise = torch.randn_like(img_t) * sigma
    out = img_t + noise
    return torch.clamp(out, 0.0, 1.0)


class IngotAugmenter:
    def __init__(
        self,
        out_size: Optional[Tuple[int, int]] = None,  # (W,H) or None
        rotation_deg: float = 10.0,
        translate_frac: float = 0.03,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        brightness_range: Tuple[float, float] = (0.90, 1.10),
        contrast_range: Tuple[float, float] = (0.90, 1.10),
        noise_prob: float = 0.7,
        noise_sigma_range: Tuple[float, float] = (0.01, 0.05),
        seed: int = 42,
    ):
        self.out_size = out_size
        self.rotation_deg = rotation_deg
        self.translate_frac = translate_frac
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_prob = noise_prob
        self.noise_sigma_range = noise_sigma_range

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()

    def _random_affine_params(self, w: int, h: int):
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)

        max_dx = int(w * self.translate_frac)
        max_dy = int(h * self.translate_frac)
        translate = (
            random.randint(-max_dx, max_dx) if max_dx > 0 else 0,
            random.randint(-max_dy, max_dy) if max_dy > 0 else 0,
        )

        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        shear = (0.0, 0.0)
        return angle, translate, scale, shear

    def _random_color_params(self):
        b = random.uniform(self.brightness_range[0], self.brightness_range[1])
        c = random.uniform(self.contrast_range[0], self.contrast_range[1])
        return b, c

    def augment_once(self, img: Image.Image) -> Image.Image:
        if self.out_size is not None:
            img = img.resize(self.out_size, resample=Image.BILINEAR)

        w, h = img.size

        # ① 幾何学変換
        angle, translate, scale, shear = self._random_affine_params(w, h)
        img = F.affine(
            img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=F.InterpolationMode.BILINEAR,
            fill=0,
        )

        # ② 輝度変換
        b, c = self._random_color_params()
        img = F.adjust_brightness(img, b)
        img = F.adjust_contrast(img, c)

        # ③ ノイズ付加
        img_t = self.to_tensor(img)
        if random.random() < self.noise_prob:
            sigma = random.uniform(self.noise_sigma_range[0], self.noise_sigma_range[1])
            img_t = add_gaussian_noise_tensor(img_t, sigma)

        return self.to_pil(img_t)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def expand_folder_each_image_to_n(
    input_dir: str,
    output_dir: str,
    n_out_each: int = 500,
    out_ext: str = ".png",
    out_size: Optional[Tuple[int, int]] = None,
    seed: int = 42,
    overwrite: bool = False,
):
    """
    input_dir内の各画像ファイルから n_out_each 枚の拡張画像を作り、
    output_dir/<stem>/ に保存。
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {in_dir}")

    paths = sorted([p for p in in_dir.iterdir() if p.is_file() and is_image_file(p)])
    if len(paths) == 0:
        raise RuntimeError(f"No image files found in: {in_dir}")

    augmenter = IngotAugmenter(out_size=out_size, seed=seed)

    print(f"Found {len(paths)} images in {in_dir.resolve()}")
    print(f"Output base dir: {out_dir.resolve()}")
    print(f"Each image -> {n_out_each} augmented images")

    for idx, img_path in enumerate(paths, start=1):
        stem = img_path.stem
        per_img_out = out_dir / stem
        per_img_out.mkdir(parents=True, exist_ok=True)

        # overwrite=Falseで既に出力があるならスキップ
        existing = list(per_img_out.glob(f"{stem}_*{out_ext}"))
        if (not overwrite) and len(existing) >= n_out_each:
            print(f"[{idx}/{len(paths)}] Skip (already exists): {img_path.name}")
            continue

        img = Image.open(img_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # 出力を一度消したい場合（overwrite=True）
        if overwrite:
            for f in per_img_out.glob(f"{stem}_*{out_ext}"):
                try:
                    f.unlink()
                except Exception:
                    pass

        for i in range(1, n_out_each + 1):
            out_img = augmenter.augment_once(img)
            out_path = per_img_out / f"{stem}_{i:04d}{out_ext}"
            out_img.save(out_path)

        print(f"[{idx}/{len(paths)}] Done: {img_path.name} -> {per_img_out.name}/ ({n_out_each} imgs)")

    print("All done.")


if __name__ == "__main__":
    # ====== ここを変更してください ======
    INPUT_DIR = r"C:\WorkSpace\Toyota\mag\DataAug\data\Fe1.5"       # 入力フォルダ（複数画像）
    OUTPUT_DIR = r"C:\WorkSpace\Toyota\mag\DataAug\output"  # 出力フォルダ
    N_OUT_EACH = 10                   # 各画像から生成する枚数

    # サイズを揃えるなら (W,H)。揃えないなら None
    OUT_SIZE = None  # 例: (512, 512)

    # 既に出力があった場合に上書きするなら True
    OVERWRITE = False

    expand_folder_each_image_to_n(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        n_out_each=N_OUT_EACH,
        out_ext=".png",
        out_size=OUT_SIZE,
        seed=42,
        overwrite=OVERWRITE,
    )
