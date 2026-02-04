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


INDEX_HTML = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AutoEncoder Anomaly Detection (Camera → API)</title>
  <style>
    :root { font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Noto Sans JP", sans-serif; }
    body { margin: 0; background: #0b0d12; color: #e9edf5; }
    header { padding: 14px 16px; border-bottom: 1px solid #1d2330; }
    h1 { font-size: 16px; margin: 0; font-weight: 700; }
    main { padding: 14px 16px; display: grid; gap: 12px; max-width: 900px; margin: 0 auto; }
    .card { background: #121725; border: 1px solid #1d2330; border-radius: 12px; padding: 12px; }
    label { display: block; font-size: 12px; color: #a9b3c7; margin-bottom: 6px; }
    input, select, button {
      width: 100%;
      box-sizing: border-box;
      padding: 10px 10px;
      border-radius: 10px;
      border: 1px solid #2a3244;
      background: #0f1422;
      color: #e9edf5;
      outline: none;
    }
    input:focus, select:focus { border-color: #4a69ff; }
    .row { display: grid; gap: 10px; grid-template-columns: 1fr; }
    @media (min-width: 720px) { .row { grid-template-columns: 1fr 1fr; } }
    .video-wrap { position: relative; overflow: hidden; border-radius: 12px; border: 1px solid #1d2330; background: #0a0c12; }
    video { width: 100%; height: auto; display: block; }
    .buttons { display: grid; gap: 10px; grid-template-columns: 1fr 1fr; }
    .buttons button { cursor: pointer; }
    button.primary { background: #2f55ff; border-color: #2f55ff; font-weight: 700; }
    button.danger { background: #ff3b3b; border-color: #ff3b3b; font-weight: 700; }
    button.secondary { background: #202a41; border-color: #2a3244; font-weight: 700; }
    .hint { font-size: 12px; color: #a9b3c7; line-height: 1.5; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .result { display: grid; gap: 8px; }
    .kv { display: grid; grid-template-columns: 160px 1fr; gap: 8px; font-size: 14px; }
    .history { display: grid; gap: 8px; }
    .history-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 6px; }
    .history-item { padding: 8px 10px; border-radius: 10px; border: 1px solid #1d2330; background: #0f1422; font-size: 12px; }
    .history-item strong { display: inline-block; min-width: 80px; }
    .toggle-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .toggle-row label { margin: 0; font-size: 12px; color: #e9edf5; display: inline-flex; gap: 6px; align-items: center; }
    .badge { display: inline-block; padding: 4px 8px; border-radius: 999px; font-size: 12px; border: 1px solid #2a3244; }
    .ok { background: rgba(25, 180, 90, 0.12); border-color: rgba(25, 180, 90, 0.35); }
    .ng { background: rgba(255, 70, 70, 0.12); border-color: rgba(255, 70, 70, 0.35); }
    .muted { color: #a9b3c7; }
    .preview-img { width: 100%; border-radius: 12px; border: 1px solid #1d2330; display: none; }
    .spinner {
      display: inline-block; width: 14px; height: 14px; border: 2px solid #2a3244; border-top-color: #e9edf5;
      border-radius: 50%; animation: spin 0.8s linear infinite; vertical-align: -2px; margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    footer { padding: 14px 16px; border-top: 1px solid #1d2330; color: #a9b3c7; font-size: 12px; }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px; font-size: 12px; border: 1px solid #2a3244; }
    .pill .dot { width: 8px; height: 8px; border-radius: 50%; background: #546075; }
    .pill.on .dot { background: #2f55ff; }
  </style>
</head>

<body>
<header>
  <h1>AutoEncoder 異常検知（スマホカメラ → FastAPI）</h1>
</header>

<main>
  <section class="card">
    <div class="row">
      <div>
        <label>API Base URL（例: <span class="mono">http://192.168.1.20:8000</span>）</label>
        <input id="apiBase" type="text" placeholder="http://localhost:8000" />
        <div class="hint" style="margin-top:6px">
          同じWi-Fiなら、FastAPIを動かしているPCのIPを入れてください。
        </div>
      </div>

      <div>
        <label>カメラ（背面優先）</label>
        <select id="cameraMode">
          <option value="environment" selected>背面（environment）</option>
          <option value="user">前面（user）</option>
        </select>
        <div class="hint" style="margin-top:6px">
          iPhoneはSafari推奨。AndroidはChromeでOKなことが多いです。
        </div>
      </div>
    </div>
  </section>

  <section class="card">
    <div class="video-wrap">
      <video id="video" playsinline autoplay muted></video>
    </div>

    <div style="margin-top:12px" class="buttons">
      <button id="startBtn" class="primary">カメラ開始</button>
      <button id="stopBtn" class="danger" disabled>カメラ停止</button>
    </div>

    <div style="margin-top:12px" class="buttons">
      <button id="snapBtn" class="secondary" disabled>手動で1回判定</button>
      <button id="retryBtn" disabled>表示リセット</button>
    </div>

    <!-- 自動判定のコントロール -->
    <div style="margin-top:12px" class="row">
      <div>
        <label>自動判定（擬似リアルタイム）</label>
        <div class="buttons">
          <button id="autoStartBtn" class="primary" disabled>自動判定開始</button>
          <button id="autoStopBtn" class="danger" disabled>自動判定停止</button>
        </div>
        <div class="hint" style="margin-top:6px">
          推論中は次のリクエストを送らないので、詰まりにくい設計です。
        </div>
      </div>
      <div>
        <label>自動判定の間隔（ms）</label>
        <input id="intervalMs" type="number" min="200" step="100" value="1000" />
        <div class="hint" style="margin-top:6px">
          まずは1000ms推奨。負荷が軽ければ500ms、重ければ2000msなど。
        </div>
      </div>
    </div>

    <div style="margin-top:12px" class="row">
      <div>
        <label>アラート</label>
        <div class="toggle-row">
          <label><input id="alertVibrate" type="checkbox" />振動</label>
          <label><input id="alertSound" type="checkbox" />サウンド</label>
          <span id="alertState" class="pill"><span class="dot"></span>アラートOFF</span>
        </div>
        <div class="hint" style="margin-top:6px">
          異常検知時にスマホへ通知します。振動は端末の設定で無効な場合があります。
        </div>
      </div>
      <div>
        <label>履歴表示</label>
        <div class="buttons">
          <button id="clearHistoryBtn" class="secondary" disabled>履歴をクリア</button>
          <button id="exportHistoryBtn" disabled>履歴をコピー</button>
        </div>
        <div class="hint" style="margin-top:6px">
          直近20件を表示します。コピーでメモや共有ができます。
        </div>
      </div>
    </div>

    <img id="preview" class="preview-img" alt="captured preview" />
    <canvas id="canvas" style="display:none;"></canvas>

    <div class="hint" style="margin-top:10px">
      ※ ローカルファイル直開きでカメラが動かない場合は、PCで簡易サーバ起動：<br/>
      <span class="mono">python -m http.server 8080</span> → スマホで <span class="mono">http://PCのIP:8080</span>
    </div>
  </section>

  <section class="card">
    <div class="row">
      <div>
        <label>threshold（任意：空なら判定しない）</label>
        <input id="threshold" type="number" step="any" placeholder="例: 0.002" />
      </div>
      <div>
        <label>image_size（モデルに合わせる：通常64）</label>
        <input id="imageSize" type="number" min="16" max="1024" value="64" />
      </div>
    </div>

    <div style="margin-top:10px">
      <label>model_path（任意：サーバのデフォルトで良ければ空）</label>
      <input id="modelPath" type="text" placeholder="例: /abs/path/best_model.onnx" />
    </div>

    <div style="margin-top:12px" class="result">
      <div class="hint">
        結果：
        <span id="busy" class="muted" style="display:none;"><span class="spinner"></span>推論中…</span>
        <span id="autoState" class="muted" style="margin-left:8px;"></span>
      </div>

      <div class="kv"><div class="muted">status</div><div id="status">-</div></div>
      <div class="kv"><div class="muted">reconstruction_error</div><div id="mse" class="mono">-</div></div>
      <div class="kv"><div class="muted">is_anomaly</div><div id="isAnomaly">-</div></div>
      <div class="kv"><div class="muted">threshold</div><div id="thrEcho" class="mono">-</div></div>

      <details>
        <summary class="muted">raw response（必要なら）</summary>
        <pre id="raw" class="mono" style="white-space:pre-wrap; word-break:break-word; margin:10px 0 0;"></pre>
      </details>
    </div>
  </section>

  <section class="card">
    <div class="history">
      <div class="hint">判定履歴</div>
      <ul id="historyList" class="history-list"></ul>
      <div id="historyEmpty" class="muted">まだ履歴がありません。</div>
    </div>
  </section>
</main>

<footer>
  サーバ側のエンドポイント：<span class="mono">POST /anomaly-score</span>, <span class="mono">GET /health</span><br/>
  うまくいかない時は、スマホで <span class="mono">/docs</span> から直接叩けるかも確認してください。
</footer>

<script>
  const $ = (id) => document.getElementById(id);

  const video = $("video");
  const canvas = $("canvas");
  const preview = $("preview");

  const apiBaseInput = $("apiBase");
  const cameraModeSel = $("cameraMode");
  const startBtn = $("startBtn");
  const stopBtn  = $("stopBtn");
  const snapBtn  = $("snapBtn");
  const retryBtn = $("retryBtn");

  const thresholdInput = $("threshold");
  const imageSizeInput = $("imageSize");
  const modelPathInput = $("modelPath");

  const intervalMsInput = $("intervalMs");
  const autoStartBtn = $("autoStartBtn");
  const autoStopBtn  = $("autoStopBtn");
  const autoStateEl  = $("autoState");

  const alertVibrateInput = $("alertVibrate");
  const alertSoundInput = $("alertSound");
  const alertStateEl = $("alertState");
  const clearHistoryBtn = $("clearHistoryBtn");
  const exportHistoryBtn = $("exportHistoryBtn");
  const historyList = $("historyList");
  const historyEmpty = $("historyEmpty");

  const statusEl = $("status");
  const mseEl = $("mse");
  const isAnomalyEl = $("isAnomaly");
  const thrEchoEl = $("thrEcho");
  const rawEl = $("raw");
  const busyEl = $("busy");

  let stream = null;
  let historyItems = [];

  // 自動判定関連
  let autoTimer = null;
  let isBusy = false;

  // デフォルト値
  apiBaseInput.value = localStorage.getItem("apiBase") || "http://localhost:8000";
  cameraModeSel.value = localStorage.getItem("cameraMode") || "environment";
  thresholdInput.value = localStorage.getItem("threshold") || "";
  imageSizeInput.value = localStorage.getItem("imageSize") || "64";
  modelPathInput.value = localStorage.getItem("modelPath") || "";
  intervalMsInput.value = localStorage.getItem("intervalMs") || "1000";
  alertVibrateInput.checked = localStorage.getItem("alertVibrate") === "true";
  alertSoundInput.checked = localStorage.getItem("alertSound") === "true";

  function setStatus(msg, kind="") {
    statusEl.textContent = msg;
    statusEl.className = kind ? `badge ${kind}` : "";
  }

  function setBusy(b) {
    isBusy = b;
    busyEl.style.display = b ? "inline" : "none";

    // UI操作制御
    snapBtn.disabled = b || !stream;
    startBtn.disabled = b;
    stopBtn.disabled = b || !stream;
    retryBtn.disabled = b;

    // 自動判定ボタン
    autoStartBtn.disabled = b || !stream || !!autoTimer;
    autoStopBtn.disabled  = b || !stream || !autoTimer;
    intervalMsInput.disabled = !!autoTimer; // 動作中は固定（暴発防止）
  }

  function setAutoState() {
    if (autoTimer) {
      autoStateEl.textContent = `自動判定: ON（${intervalMsInput.value}ms）`;
    } else {
      autoStateEl.textContent = `自動判定: OFF`;
    }
  }

  function setAlertState() {
    const enabled = alertVibrateInput.checked || alertSoundInput.checked;
    alertStateEl.classList.toggle("on", enabled);
    alertStateEl.innerHTML = `<span class="dot"></span>${enabled ? "アラートON" : "アラートOFF"}`;
  }

  function saveHistory() {
    localStorage.setItem("historyItems", JSON.stringify(historyItems));
  }

  function loadHistory() {
    try {
      const raw = localStorage.getItem("historyItems");
      historyItems = raw ? JSON.parse(raw) : [];
    } catch (e) {
      historyItems = [];
    }
  }

  function renderHistory() {
    historyList.innerHTML = "";
    if (historyItems.length === 0) {
      historyEmpty.style.display = "block";
      clearHistoryBtn.disabled = true;
      exportHistoryBtn.disabled = true;
      return;
    }

    historyEmpty.style.display = "none";
    historyItems.forEach((item) => {
      const li = document.createElement("li");
      li.className = "history-item";
      li.innerHTML = `<strong>${item.label}</strong>${item.mse} (thr: ${item.threshold})<br/><span class="muted">${item.time}</span>`;
      historyList.appendChild(li);
    });
    clearHistoryBtn.disabled = false;
    exportHistoryBtn.disabled = false;
  }

  function pushHistory({ mse, threshold, isAnomaly }) {
    const label = isAnomaly === null ? "判定なし" : (isAnomaly ? "異常" : "正常");
    historyItems.unshift({
      label,
      mse: typeof mse === "number" ? mse.toPrecision(6) : String(mse),
      threshold: threshold === null || threshold === undefined ? "-" : String(threshold),
      time: new Date().toLocaleString(),
    });
    historyItems = historyItems.slice(0, 20);
    saveHistory();
    renderHistory();
  }

  function vibrateAlert() {
    if (!alertVibrateInput.checked) return;
    if ("vibrate" in navigator) {
      navigator.vibrate([120, 80, 120]);
    }
  }

  function soundAlert() {
    if (!alertSoundInput.checked) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 880;
      gain.gain.value = 0.05;
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      setTimeout(() => {
        osc.stop();
        ctx.close();
      }, 200);
    } catch (e) {
      console.warn("audio alert failed", e);
    }
  }

  function saveSettings() {
    localStorage.setItem("apiBase", apiBaseInput.value.trim());
    localStorage.setItem("cameraMode", cameraModeSel.value);
    localStorage.setItem("threshold", thresholdInput.value);
    localStorage.setItem("imageSize", imageSizeInput.value);
    localStorage.setItem("modelPath", modelPathInput.value.trim());
    localStorage.setItem("intervalMs", intervalMsInput.value);
    localStorage.setItem("alertVibrate", alertVibrateInput.checked);
    localStorage.setItem("alertSound", alertSoundInput.checked);
  }

  async function startCamera() {
    saveSettings();
    const facingMode = cameraModeSel.value;

    await stopCamera();

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      video.srcObject = stream;
      await video.play();

      startBtn.disabled = true;
      stopBtn.disabled = false;
      snapBtn.disabled = false;
      retryBtn.disabled = false;

      autoStartBtn.disabled = false; // カメラ開始したら自動判定開始可能

      preview.style.display = "none";
      setStatus("camera ready", "ok");
      setBusy(false);
      setAutoState();
    } catch (err) {
      console.error(err);
      stream = null;
      setStatus("camera error: " + err, "ng");
      setBusy(false);
      setAutoState();
    }
  }

  async function stopCamera() {
    stopAuto(); // カメラ止めるなら自動判定も止める

    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    video.srcObject = null;

    startBtn.disabled = false;
    stopBtn.disabled = true;
    snapBtn.disabled = true;

    autoStartBtn.disabled = true;
    autoStopBtn.disabled = true;

    setAutoState();
  }

  function buildUrl() {
    const base = apiBaseInput.value.trim().replace(/\/+$/, "");
    const url = new URL(base + "/anomaly-score");

    const thr = thresholdInput.value.trim();
    const imageSize = imageSizeInput.value.trim();
    const modelPath = modelPathInput.value.trim();

    if (thr !== "") url.searchParams.set("threshold", thr);
    if (imageSize !== "") url.searchParams.set("image_size", imageSize);
    if (modelPath !== "") url.searchParams.set("model_path", modelPath);

    return url.toString();
  }

  function captureJpegBlob(quality = 0.85) {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) throw new Error("Video not ready");

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h);

    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), "image/jpeg", quality);
    });
  }

  async function pingHealth() {
    const base = apiBaseInput.value.trim().replace(/\/+$/, "");
    try {
      const r = await fetch(base + "/health", { method: "GET" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const j = await r.json();
      return j?.status === "ok";
    } catch (e) {
      return false;
    }
  }

  async function inferOnce({ showPreview = false } = {}) {
    if (!stream) return;
    if (isBusy) return; // 推論中はスキップ（連打/多重送信防止）

    saveSettings();
    setBusy(true);
    setStatus("processing…");
    mseEl.textContent = "-";
    isAnomalyEl.textContent = "-";
    thrEchoEl.textContent = "-";
    rawEl.textContent = "";

    try {
      const ok = await pingHealth();
      if (!ok) {
        throw new Error("API /health に接続できません。API Base URL を確認してください。");
      }

      const blob = await captureJpegBlob(0.85);

      if (showPreview) {
        preview.src = URL.createObjectURL(blob);
        preview.style.display = "block";
      }

      const form = new FormData();
      form.append("file", blob, "capture.jpg");

      const url = buildUrl();
      const res = await fetch(url, { method: "POST", body: form });
      const text = await res.text();

      if (!res.ok) {
        throw new Error(`API error (HTTP ${res.status}): ${text}`);
      }

      const data = JSON.parse(text);

      const mse = data.reconstruction_error;
      const thr = data.threshold;
      const isAnomaly = data.is_anomaly;

      mseEl.textContent = (typeof mse === "number") ? mse.toPrecision(6) : String(mse);
      thrEchoEl.textContent = (thr === null || thr === undefined) ? "-" : String(thr);

      if (isAnomaly === null || isAnomaly === undefined) {
        isAnomalyEl.textContent = "-（threshold未指定）";
        setStatus("done", "ok");
      } else if (isAnomaly) {
        isAnomalyEl.textContent = "true";
        setStatus("ANOMALY", "ng");
        vibrateAlert();
        soundAlert();
      } else {
        isAnomalyEl.textContent = "false";
        setStatus("NORMAL", "ok");
      }

      rawEl.textContent = JSON.stringify(data, null, 2);
      pushHistory({ mse, threshold: thr, isAnomaly });
    } catch (err) {
      console.error(err);
      setStatus(String(err), "ng");
      rawEl.textContent = String(err);
    } finally {
      setBusy(false);
      setAutoState();
    }
  }

  function startAuto() {
    if (!stream) return;
    if (autoTimer) return;

    const interval = Number(intervalMsInput.value);
    const safeInterval = Number.isFinite(interval) ? Math.max(200, interval) : 1000;
    intervalMsInput.value = String(safeInterval);
    saveSettings();

    // まず1回すぐに実行（体感を良くする）
    inferOnce({ showPreview: false });

    autoTimer = setInterval(() => {
      inferOnce({ showPreview: false });
    }, safeInterval);

    setAutoState();
    setBusy(isBusy);
    setStatus("auto running", "ok");
  }

  function stopAuto() {
    if (autoTimer) {
      clearInterval(autoTimer);
      autoTimer = null;
    }
    setAutoState();
    setBusy(isBusy);
  }

  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", stopCamera);
  snapBtn.addEventListener("click", () => inferOnce({ showPreview: true }));
  retryBtn.addEventListener("click", () => { setStatus("-"); rawEl.textContent = ""; });

  autoStartBtn.addEventListener("click", startAuto);
  autoStopBtn.addEventListener("click", stopAuto);

  alertVibrateInput.addEventListener("change", () => { saveSettings(); setAlertState(); });
  alertSoundInput.addEventListener("change", () => { saveSettings(); setAlertState(); });

  clearHistoryBtn.addEventListener("click", () => {
    historyItems = [];
    saveHistory();
    renderHistory();
  });

  exportHistoryBtn.addEventListener("click", async () => {
    const lines = historyItems.map((item) => `${item.time}\t${item.label}\t${item.mse}\tthr=${item.threshold}`);
    const text = lines.join("\n");
    try {
      await navigator.clipboard.writeText(text);
      setStatus("履歴をクリップボードにコピーしました", "ok");
    } catch (err) {
      setStatus("コピーに失敗しました", "ng");
    }
  });

  // 対応チェック
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus("このブラウザはカメラAPIに対応していません", "ng");
    startBtn.disabled = true;
  } else {
    setStatus("ready");
  }
  setAutoState();
  setAlertState();
  loadHistory();
  renderHistory();

  // ページを離れる時に停止
  window.addEventListener("beforeunload", () => {
    stopAuto();
    stopCamera();
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
