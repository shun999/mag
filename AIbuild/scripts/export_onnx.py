"""
AutoEncoderモデルをONNXに変換するスクリプト
"""

import argparse
from pathlib import Path

import torch

from model import create_model


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get("args", {})

    model = create_model(
        input_channels=args.get("input_channels", 3),
        latent_dim=args.get("latent_dim", 128),
        device=device,
        use_skip=args.get("use_skip", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, args


def export_onnx(checkpoint_path, output_path, opset=17, device="cpu"):
    model, args = load_model(checkpoint_path, device=device)

    image_size = args.get("image_size", [64, 64])
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        height, width = int(image_size[0]), int(image_size[1])
    else:
        height, width = 64, 64

    input_channels = args.get("input_channels", 3)

    dummy_input = torch.randn(1, input_channels, height, width, device=device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["reconstructed", "latent"],
        dynamic_axes={
            "input": {0: "batch"},
            "reconstructed": {0: "batch"},
            "latent": {0: "batch"},
        },
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export AutoEncoder to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="学習済みモデルのチェックポイントパス",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力するONNXファイルパス",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="デバイス (cpu/cuda)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = export_onnx(
        args.checkpoint, args.output, opset=args.opset, device=device
    )
    print(f"ONNX exported to: {output_path}")


if __name__ == "__main__":
    main()
