#!/usr/bin/env bash
# Sasage AI 開発用起動スクリプト: FastAPIバックエンド + Streamlit UI を同時起動する
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR/backend"
uvicorn app.main:app --reload --port 8000 &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

cd "$ROOT_DIR"
SASAGE_API_BASE_URL="http://localhost:8000" streamlit run ui/streamlit_app.py
