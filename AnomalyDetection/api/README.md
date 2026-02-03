# AutoEncoder ONNX API

## 起動

```bash
uvicorn AnomalyDetection.api.app:app --host 0.0.0.0 --port 8000
```

## 推論

```bash
curl -X POST "http://localhost:8000/anomaly-score" \
  -F "file=@/path/to/image.png"
```

閾値を指定して異常判定も返す場合:

```bash
curl -X POST "http://localhost:8000/anomaly-score?threshold=0.01" \
  -F "file=@/path/to/image.png"
```

モデルパスや画像サイズを変える場合:

```bash
curl -X POST "http://localhost:8000/anomaly-score?model_path=/path/to/model.onnx&image_size=64" \
  -F "file=@/path/to/image.png"
```
