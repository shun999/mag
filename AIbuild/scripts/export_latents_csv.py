"""
AutoEncoderの潜在ベクトルをCSVに書き出すスクリプト
指定ディレクトリ内の画像をエンコードしてCSVへ保存
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from model import create_model


def load_model(checkpoint_path, device="cuda"):
    """学習済みモデルを読み込む"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})

    model = create_model(
        input_channels=args.get("input_channels", 3),
        latent_dim=args.get("latent_dim", 128),
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, args


def encode_image(model, image_path, image_size=(64, 64), device="cuda"):
    """単一画像を潜在ベクトルへ変換"""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
        ]
    )

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, latent = model(img_tensor)

    return latent.cpu().numpy().squeeze(0)


def collect_image_paths(image_dir):
    image_paths = []
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        image_paths.extend(image_dir.rglob(f"*{ext}"))
        image_paths.extend(image_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(image_paths))


def main():
    parser = argparse.ArgumentParser(
        description="指定ディレクトリ内の画像を潜在ベクトルに変換してCSVへ保存"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="学習済みモデルのチェックポイントパス",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="対象画像が入ったディレクトリ",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./latents.csv",
        help="出力CSVファイルパス",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="処理する画像数の上限（省略時はすべて）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="デバイス (auto/cuda/cpu)",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    model, model_args = load_model(args.checkpoint, device=device)
    img_size = model_args.get("image_size", [64, 64])
    image_size = tuple(img_size) if isinstance(img_size, (list, tuple)) else (64, 64)
    latent_dim = int(model_args.get("latent_dim", 128))
    print(f"Model loaded. Image size: {image_size}, latent_dim: {latent_dim}")

    image_dir = Path(args.image_dir)
    image_paths = collect_image_paths(image_dir)
    if args.limit:
        image_paths = image_paths[: args.limit]

    print(f"Found {len(image_paths)} images.")
    if not image_paths:
        print("No images found. Exiting.")
        return

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    header = ["image_path"] + [f"latent_{i}" for i in range(latent_dim)]

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for idx, img_path in enumerate(image_paths, 1):
            print(f"[{idx}/{len(image_paths)}] Encoding: {img_path}")
            try:
                latent = encode_image(model, img_path, image_size, device)
                latent_flat = np.asarray(latent).reshape(-1).tolist()
                if len(latent_flat) != latent_dim:
                    print(
                        f"Warning: latent dim mismatch for {img_path} "
                        f"({len(latent_flat)} != {latent_dim}). Skipping."
                    )
                    continue
                writer.writerow([str(img_path)] + latent_flat)
            except Exception as exc:
                print(f"Error processing {img_path}: {exc}")
                continue

    print(f"Saved CSV to: {output_csv}")


if __name__ == "__main__":
    main()
