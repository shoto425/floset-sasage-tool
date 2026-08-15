"""Sasage AI FastAPI エントリポイント。"""
from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import products

app = FastAPI(
    title="Sasage AI",
    description="1本の動画から平置き画像・採寸・状態検品・原稿・ZIPをワンストップ生成する古着ささげ効率化ツール",
    version="0.1.0",
)

app.include_router(products.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
