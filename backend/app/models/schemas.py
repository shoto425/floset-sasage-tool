"""Sasage AI の中核データモデル。

Phase 1 (静止画中心) のパイプライン全体で受け渡しされる型をここに集約する。
Phase 2 (動画・複数アングル対応) でもこれらのモデルを拡張して再利用する想定。
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class GarmentCategory(str, Enum):
    """採寸・検品のロジックを出し分けるための衣類カテゴリ。"""

    TOPS = "tops"
    OUTER = "outer"
    BOTTOMS = "bottoms"
    DRESS = "dress"
    OTHER = "other"


class MeasurementPoint(BaseModel):
    """採寸1項目分のデータ。"""

    label: str = Field(..., description="採寸項目名（例: 肩幅）")
    value_cm: Optional[float] = Field(
        None, description="推定値(cm)。マスク検出失敗時は None。"
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="推定の信頼度(0-1)。ヒューリスティックの粗さを反映。"
    )
    method: str = Field(
        "heuristic_mask_ratio",
        description="推定方法。将来 'mediapipe_keypoint' 等に置き換わる。",
    )


class MeasurementSet(BaseModel):
    """1商品分の採寸データ一式。"""

    category: GarmentCategory
    reference_label: str = Field(..., description="基準実測値として使った項目名（例: 着丈）")
    reference_value_cm: float = Field(..., description="ユーザーが実測して入力した基準値(cm)")
    scale_cm_per_px: float = Field(..., description="基準値から算出した cm/px スケール")
    points: list[MeasurementPoint] = Field(default_factory=list)

    def get(self, label: str) -> Optional[MeasurementPoint]:
        return next((p for p in self.points if p.label == label), None)


class ConditionSeverity(str, Enum):
    LOW = "low"       # 軽微（色落ち等、目立たない）
    MEDIUM = "medium"  # 中程度（要説明）
    HIGH = "high"     # 重度（要写真添付・価格反映レベル）


class ConditionIssue(BaseModel):
    """検出された状態・傷の1件分。"""

    description: str = Field(..., description="例: '袖口のほつれ'")
    severity: ConditionSeverity
    location_hint: str = Field(..., description="例: '左袖口'")
    recommend_closeup: bool = Field(
        True, description="該当箇所の接写を推奨するか"
    )
    bbox_px: Optional[tuple[int, int, int, int]] = Field(
        None, description="元画像上のバウンディングボックス (x, y, w, h)"
    )
    detected_by: str = Field(
        "heuristic_cv", description="検出方法。'heuristic_cv' または 'llm_vision'"
    )


class GeneratedCopy(BaseModel):
    """原稿生成の結果。"""

    instagram_caption: str
    ec_description: str
    hashtags: list[str] = Field(default_factory=list)
    generated_by: str = Field("llm", description="'llm' または 'template_fallback'")


class FlatLayImage(BaseModel):
    """生成された平置き画像1枚分のメタデータ。"""

    file_name: str
    source_frame_index: int = Field(..., description="元動画内でのフレーム位置")
    sharpness_score: float
    width: int
    height: int
    is_primary: bool = Field(False, description="サムネイル代表カットかどうか")


class ProductInput(BaseModel):
    """パイプライン実行時にユーザーが与える入力。"""

    product_name: str
    category: GarmentCategory = GarmentCategory.TOPS
    brand: Optional[str] = None
    reference_measurement_label: str = Field(
        "着丈", description="基準として実測した採寸項目名"
    )
    reference_measurement_cm: float = Field(
        ..., gt=0, description="基準実測値(cm)。この1点から他の採寸を比率推定する。"
    )
    notes: Optional[str] = Field(None, description="出品者からの補足メモ")


class ProductResult(BaseModel):
    """パイプライン処理結果一式。ZIP生成・API応答の両方で使う。"""

    product_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    input: ProductInput
    flat_lay_images: list[FlatLayImage] = Field(default_factory=list)
    measurements: Optional[MeasurementSet] = None
    condition_issues: list[ConditionIssue] = Field(default_factory=list)
    generated_copy: Optional[GeneratedCopy] = None
    output_dir: str
    zip_path: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
