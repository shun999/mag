"""
ONNX版AutoEncoder異常検知API
"""

from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
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


INDEX_HTML = """<!doctype html>
<html lang="ja">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AutoEncoder Anomaly Detection</title>
    <style>
      :root {
        color-scheme: light dark;
        font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      }
      body {
        margin: 0;
        padding: 32px;
        background: #0f172a;
        color: #e2e8f0;
      }
      .card {
        max-width: 720px;
        margin: 0 auto;
        background: #111827;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.5);
      }
      h1 {
        font-size: 1.6rem;
        margin-bottom: 0.5rem;
      }
      p {
        color: #cbd5f5;
        line-height: 1.5;
      }
      label {
        display: block;
        font-weight: 600;
        margin: 16px 0 8px;
      }
      input[type="file"],
      input[type="number"],
      input[type="text"] {
        width: 100%;
        padding: 10px 12px;
        border-radius: 10px;
        border: 1px solid #334155;
        background: #0b1120;
        color: #e2e8f0;
      }
      button {
        margin-top: 20px;
        padding: 12px 18px;
        border-radius: 999px;
        border: none;
        background: #38bdf8;
        color: #0f172a;
        font-weight: 700;
        cursor: pointer;
      }
      button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      .results {
        margin-top: 24px;
        padding: 16px;
        background: #0b1120;
        border-radius: 12px;
        border: 1px solid #1e293b;
      }
      .results pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        color: #e2e8f0;
      }
      .status {
        margin-top: 12px;
        font-size: 0.95rem;
        color: #94a3b8;
      }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>AutoEncoder 異常検知デモ</h1>
      <p>画像をアップロードして再構成誤差を確認します。しきい値を入れると異常判定も表示します。</p>
      <form id="anomaly-form">
        <label for="image">画像ファイル</label>
        <input id="image" name="image" type="file" accept="image/*" required />

        <label for="threshold">しきい値 (任意)</label>
        <input id="threshold" name="threshold" type="number" step="0.0001" placeholder="例: 0.02" />

        <label for="image-size">画像サイズ (正方形)</label>
        <input id="image-size" name="image-size" type="number" value="64" min="16" max="512" />

        <label for="model-path">モデルパス</label>
        <input id="model-path" name="model-path" type="text" />

        <button type="submit" id="submit-btn">解析する</button>
      </form>

      <div class="status" id="status"></div>
      <div class="results" id="results" hidden>
        <pre id="results-json"></pre>
      </div>
    </div>

    <script>
      const form = document.getElementById("anomaly-form");
      const statusEl = document.getElementById("status");
      const resultsWrap = document.getElementById("results");
      const resultsJson = document.getElementById("results-json");
      const submitBtn = document.getElementById("submit-btn");
      const modelPathInput = document.getElementById("model-path");
      modelPathInput.value = new URL("/AutoEncoder/output/onnx/best_model.onnx", window.location.origin).pathname;

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        statusEl.textContent = "";
        resultsWrap.hidden = true;
        submitBtn.disabled = true;

        const imageInput = document.getElementById("image");
        if (!imageInput.files.length) {
          statusEl.textContent = "画像を選択してください。";
          submitBtn.disabled = false;
          return;
        }

        const formData = new FormData();
        formData.append("file", imageInput.files[0]);

        const thresholdValue = document.getElementById("threshold").value;
        const imageSizeValue = document.getElementById("image-size").value || "64";
        const modelPathValue = modelPathInput.value;

        const params = new URLSearchParams();
        if (thresholdValue) {
          params.append("threshold", thresholdValue);
        }
        if (imageSizeValue) {
          params.append("image_size", imageSizeValue);
        }
        if (modelPathValue) {
          params.append("model_path", modelPathValue);
        }

        statusEl.textContent = "解析中...";

        try {
          const response = await fetch(`/anomaly-score?${params.toString()}`, {
            method: "POST",
            body: formData,
          });
          const data = await response.json();
          if (!response.ok) {
            throw new Error(data.detail || "エラーが発生しました。");
          }
          resultsJson.textContent = JSON.stringify(data, null, 2);
          resultsWrap.hidden = false;
          statusEl.textContent = "完了しました。";
        } catch (error) {
          statusEl.textContent = `失敗しました: ${error.message}`;
        } finally {
          submitBtn.disabled = false;
        }
      });
    </script>
  </body>
</html>
"""


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


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(INDEX_HTML)


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
