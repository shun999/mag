"""
フォルダ内の全画像を対象に、
「①幾何学変換＋②輝度・色変換＋③ぼかし＋④ノイズ付加＋⑤ランダム消去」を
組み合わせて各画像からN枚ずつ拡張して保存するスクリプト。

- 入力: input_dir（画像が入ったフォルダ）
- 出力: output_dir / <元ファイル名>/ に N枚保存
  例: output/normal_001/normal_001_0001.png ... normal_001_0500.png

依存:
  pip install pillow torchvision torch numpy
"""

import argparse
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

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
        # ① 幾何学変換パラメータ
        rotation_deg: float = 30.0,
        translate_frac: float = 0.08,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        shear_range: Tuple[float, float] = (-5.0, 5.0),
        flip_h_prob: float = 0.5,
        flip_v_prob: float = 0.5,
        perspective_prob: float = 0.2,
        perspective_scale: float = 0.1,
        # ② 輝度・色変換パラメータ
        brightness_range: Tuple[float, float] = (0.75, 1.25),
        contrast_range: Tuple[float, float] = (0.75, 1.25),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: float = 0.05,
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        # ③ ぼかしパラメータ
        blur_prob: float = 0.3,
        blur_kernel_range: Tuple[int, int] = (3, 7),
        blur_sigma_range: Tuple[float, float] = (0.1, 2.0),
        # ④ ノイズパラメータ
        noise_prob: float = 0.7,
        noise_sigma_range: Tuple[float, float] = (0.005, 0.08),
        # ⑤ ランダム消去パラメータ
        erasing_prob: float = 0.15,
        erasing_scale: Tuple[float, float] = (0.02, 0.10),
        seed: int = 42,
    ):
        self.out_size = out_size
        # ①
        self.rotation_deg = rotation_deg
        self.translate_frac = translate_frac
        self.scale_range = scale_range
        self.shear_range = shear_range
        self.flip_h_prob = flip_h_prob
        self.flip_v_prob = flip_v_prob
        self.perspective_prob = perspective_prob
        self.perspective_scale = perspective_scale
        # ②
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.gamma_range = gamma_range
        # ③
        self.blur_prob = blur_prob
        self.blur_kernel_range = blur_kernel_range
        self.blur_sigma_range = blur_sigma_range
        # ④
        self.noise_prob = noise_prob
        self.noise_sigma_range = noise_sigma_range
        # ⑤
        self.erasing_prob = erasing_prob
        self.erasing_scale = erasing_scale

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
        shear = (
            random.uniform(self.shear_range[0], self.shear_range[1]),
            random.uniform(self.shear_range[0], self.shear_range[1]),
        )
        return angle, translate, scale, shear

    def _random_color_params(self):
        b = random.uniform(self.brightness_range[0], self.brightness_range[1])
        c = random.uniform(self.contrast_range[0], self.contrast_range[1])
        s = random.uniform(self.saturation_range[0], self.saturation_range[1])
        h = random.uniform(-self.hue_range, self.hue_range)
        return b, c, s, h

    def augment_once(self, img: Image.Image) -> Image.Image:
        if self.out_size is not None:
            img = img.resize(self.out_size, resample=Image.BILINEAR)

        w, h = img.size

        # ① 幾何学変換
        # フリップ
        if random.random() < self.flip_h_prob:
            img = F.hflip(img)
        if random.random() < self.flip_v_prob:
            img = F.vflip(img)

        # アフィン変換
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

        # 透視変換
        if random.random() < self.perspective_prob:
            img = F.perspective(
                img,
                startpoints=[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                endpoints=[
                    [int(random.uniform(0, w * self.perspective_scale)),
                     int(random.uniform(0, h * self.perspective_scale))],
                    [int(random.uniform(w * (1 - self.perspective_scale), w - 1)),
                     int(random.uniform(0, h * self.perspective_scale))],
                    [int(random.uniform(w * (1 - self.perspective_scale), w - 1)),
                     int(random.uniform(h * (1 - self.perspective_scale), h - 1))],
                    [int(random.uniform(0, w * self.perspective_scale)),
                     int(random.uniform(h * (1 - self.perspective_scale), h - 1))],
                ],
                interpolation=F.InterpolationMode.BILINEAR,
                fill=0,
            )

        # ② 輝度・色変換
        b, c, s, hue = self._random_color_params()
        img = F.adjust_brightness(img, b)
        img = F.adjust_contrast(img, c)
        img = F.adjust_saturation(img, s)
        img = F.adjust_hue(img, hue)

        # ガンマ補正
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        img = F.adjust_gamma(img, gamma)

        # ③ ガウシアンぼかし
        if random.random() < self.blur_prob:
            kernel_size = random.choice(
                range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2)
            )
            sigma = random.uniform(self.blur_sigma_range[0], self.blur_sigma_range[1])
            img = F.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=sigma)

        # ④ ノイズ付加
        img_t = self.to_tensor(img)
        if random.random() < self.noise_prob:
            sigma = random.uniform(self.noise_sigma_range[0], self.noise_sigma_range[1])
            img_t = add_gaussian_noise_tensor(img_t, sigma)

        # ⑤ ランダム消去
        if random.random() < self.erasing_prob:
            img_t = T.RandomErasing(
                p=1.0,
                scale=self.erasing_scale,
                ratio=(0.3, 3.3),
                value=0,
            )(img_t)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Image augmentation for training data expansion")
    parser.add_argument("--input_dir", type=str, default="DataAug/data",
                        help="入力画像フォルダ")
    parser.add_argument("--output_dir", type=str, default="DataAug/output",
                        help="出力フォルダ")
    parser.add_argument("--n_each", type=int, default=50,
                        help="各画像から生成する拡張枚数")
    parser.add_argument("--image_size", type=int, nargs=2, default=None,
                        help="出力画像サイズ [W H]。未指定で元サイズ維持")
    parser.add_argument("--seed", type=int, default=42,
                        help="乱数シード")
    parser.add_argument("--overwrite", action="store_true",
                        help="既存出力を上書き")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    out_size = tuple(args.image_size) if args.image_size else None

    expand_folder_each_image_to_n(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_out_each=args.n_each,
        out_ext=".png",
        out_size=out_size,
        seed=args.seed,
        overwrite=args.overwrite,
    )
