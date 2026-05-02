"""
ONNX版AutoEncoder異常検知API (AWS EC2デプロイ用)
起動時に S3 からモデルファイルをダウンロードする。
"""

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from scipy.ndimage import gaussian_filter
import onnxruntime as ort


# ============================================================
# 設定 (環境変数)
# ============================================================
S3_BUCKET = os.environ.get("S3_BUCKET", "your-bucket-name")
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY", "models/best_model.onnx")
S3_STATS_KEY = os.environ.get("S3_STATS_KEY", "models/detection_stats.npz")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
DEFAULT_IMAGE_SIZE = (64, 64)

MODEL_PATH = MODEL_DIR / "best_model.onnx"
STATS_PATH = MODEL_DIR / "detection_stats.npz"


# ============================================================
# S3 からモデルファイルをダウンロード
# ============================================================
def _download_from_s3():
    """S3 からモデルファイルをダウンロードする (存在しない場合のみ)。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    for local_path, s3_key in [(MODEL_PATH, S3_MODEL_KEY), (STATS_PATH, S3_STATS_KEY)]:
        if local_path.exists():
            print(f"[S3] Already exists: {local_path}")
            continue
        print(f"[S3] Downloading s3://{S3_BUCKET}/{s3_key} -> {local_path}")
        s3.download_file(S3_BUCKET, s3_key, str(local_path))
        print(f"[S3] Done: {local_path}")


# ============================================================
# FastAPI lifespan (起動時に S3 ダウンロード)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    _download_from_s3()
    yield


app = FastAPI(title="AutoEncoder Anomaly Detection API", lifespan=lifespan)


# ============================================================
# 画像読み込み
# ============================================================
def _load_image(file_bytes: bytes, image_size):
    try:
        img = Image.open(BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(image_size)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ============================================================
# SSIM (NumPy/scipy版)
# ============================================================
def _compute_ssim(
    x: np.ndarray, y: np.ndarray,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2, C2: float = 0.03 ** 2,
) -> float:
    """Compute mean SSIM between two [1, C, H, W] float32 arrays."""
    x = x[0]
    y = y[0]
    ssim_per_channel = []
    for c in range(x.shape[0]):
        xc, yc = x[c], y[c]
        mu_x = gaussian_filter(xc, sigma=sigma)
        mu_y = gaussian_filter(yc, sigma=sigma)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x2 = gaussian_filter(xc ** 2, sigma=sigma) - mu_x2
        sigma_y2 = gaussian_filter(yc ** 2, sigma=sigma) - mu_y2
        sigma_xy = gaussian_filter(xc * yc, sigma=sigma) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = num / den
        ssim_per_channel.append(ssim_map.mean())

    return float(np.mean(ssim_per_channel))


# ============================================================
# Mahalanobis距離
# ============================================================
def _mahalanobis_distance(
    z: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray,
) -> float:
    """Compute Mahalanobis distance for a single latent vector."""
    diff = z - mean
    return float(np.sqrt(diff @ inv_cov @ diff))


# ============================================================
# モデル・統計量の読み込み (キャッシュ)
# ============================================================
@lru_cache(maxsize=1)
def get_session(model_path: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    return session, input_name, output_names


@lru_cache(maxsize=1)
def get_detection_stats(stats_path: str) -> dict:
    if not Path(stats_path).exists():
        raise FileNotFoundError(f"Detection stats not found: {stats_path}")
    data = np.load(stats_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


# ============================================================
# エンドポイント
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/anomaly-score")
async def anomaly_score(
    file: UploadFile = File(...),
    image_size: int = DEFAULT_IMAGE_SIZE[0],
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    try:
        session, input_name, output_names = get_session(str(MODEL_PATH))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        stats = get_detection_stats(str(STATS_PATH))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # 推論
    file_bytes = await file.read()
    input_tensor = _load_image(file_bytes, (image_size, image_size))
    reconstructed, latent = session.run(output_names, {input_name: input_tensor})

    # 各指標を計算
    mse = float(np.mean((reconstructed - input_tensor) ** 2))
    ssim_val = _compute_ssim(input_tensor, reconstructed)
    mahal_dist = _mahalanobis_distance(
        latent[0], stats['latent_mean'], stats['inv_cov'],
    )

    # 正規化 → Ensemble Score
    eps = 1e-8
    norm_mse = (mse - float(stats['mse_mean'])) / (float(stats['mse_std']) + eps)
    norm_ssim = -(ssim_val - float(stats['ssim_mean'])) / (float(stats['ssim_std']) + eps)
    norm_mahal = (mahal_dist - float(stats['mahal_mean'])) / (float(stats['mahal_std']) + eps)
    ensemble_score = (norm_mse + norm_ssim + norm_mahal) / 3.0

    threshold = float(stats['ensemble_threshold'])

    response = {
        "ensemble_score": ensemble_score,
        "is_anomaly": bool(ensemble_score > threshold),
        "threshold": threshold,
        "reconstruction_error": mse,
        "ssim_score": ssim_val,
        "mahalanobis_distance": mahal_dist,
        "latent_vector": latent[0].tolist(),
    }

    return JSONResponse(content=response)
