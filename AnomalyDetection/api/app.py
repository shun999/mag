"""
ONNX版AutoEncoder異常検知API
"""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import onnxruntime as ort


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "AutoEncoder"
    / "output"
    / "onnx"
    / "best_model.onnx"
)
DEFAULT_IMAGE_SIZE = (64, 64)

app = FastAPI(title="AutoEncoder Anomaly Detection API")


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


@lru_cache(maxsize=1)
def get_session(model_path: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    return session, input_name, output_names


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/anomaly-score")
async def anomaly_score(
    file: UploadFile = File(...),
    threshold: float | None = None,
    model_path: str = str(DEFAULT_MODEL_PATH),
    image_size: int = DEFAULT_IMAGE_SIZE[0],
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")

    try:
        session, input_name, output_names = get_session(model_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    file_bytes = await file.read()
    input_tensor = _load_image(file_bytes, (image_size, image_size))

    reconstructed, latent = session.run(output_names, {input_name: input_tensor})

    mse = float(np.mean((reconstructed - input_tensor) ** 2))
    response = {
        "reconstruction_error": mse,
        "threshold": threshold,
        "is_anomaly": bool(mse > threshold) if threshold is not None else None,
        "latent_vector": latent[0].tolist(),
    }

    return JSONResponse(content=response)
