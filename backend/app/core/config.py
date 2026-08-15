"""アプリケーション設定。

環境変数 (.env) から読み込む。API キーが無くてもヒューリスティック実装のみで
動作するようにデフォルトを倒してある。
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SASAGE_", extra="ignore")

    # ストレージ
    data_dir: Path = Path("data")
    uploads_dir_name: str = "uploads"
    outputs_dir_name: str = "outputs"

    # フレーム抽出
    max_candidate_frames: int = 60
    max_selected_frames: int = 6
    frame_dedupe_hash_distance: int = 8  # pHashの距離がこれ未満なら「似た構図」とみなす

    # 背景除去
    background_removal_backend: str = "rembg"  # "rembg" | "grabcut"
    output_canvas_size: int = 1600
    output_padding_ratio: float = 0.08  # 余白比率

    # 状態検品
    condition_backend: str = "heuristic"  # "heuristic" | "llm_vision"
    condition_edge_density_zscore_threshold: float = 2.0

    # 原稿生成 / LLM
    llm_backend: str = "template_fallback"  # "claude" | "template_fallback"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / self.uploads_dir_name

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / self.outputs_dir_name


settings = Settings()
