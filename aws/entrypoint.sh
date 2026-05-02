#!/bin/bash
set -e

echo "Starting Anomaly Detection API..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
