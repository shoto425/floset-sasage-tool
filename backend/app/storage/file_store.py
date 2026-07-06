"""ローカルファイルストレージの薄い抽象化。

Phase 2 で S3 / GCS に差し替える際、このモジュールのインターフェースだけを
実装し直せばパイプライン側のコードは変更不要にする狙い。
"""
from __future__ import annotations

import uuid
from pathlib import Path

from app.core.config import settings


class FileStore:
    """商品(product)単位のディレクトリを払い出すローカルストレージ。"""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or settings.outputs_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    def new_product_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def product_dir(self, product_id: str) -> Path:
        path = self.base_dir / product_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def images_dir(self, product_id: str) -> Path:
        path = self.product_dir(product_id) / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def upload_path(self, product_id: str, filename: str) -> Path:
        path = settings.uploads_dir / product_id
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def zip_path(self, product_id: str) -> Path:
        return self.product_dir(product_id) / f"{product_id}.zip"


file_store = FileStore()
