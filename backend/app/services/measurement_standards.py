"""カテゴリ別の標準採寸項目定義。

古着ECで一般的に求められる採寸項目を、フクダス等の既存ささげ代行サービスや
主要古着ECサイトの表記に合わせて標準化したもの。`MeasurementService` は
ここで定義された `label` をそのまま `MeasurementPoint.label` に使う。

`phase1_supported=False` の項目は Phase 1 のマスク形状ヒューリスティックでは
安定して推定できないため、`MeasurementPoint(value_cm=None, method="not_supported_phase1")`
として結果に含め、「非対応であること」自体を採寸データ上で明示する
（項目が黙って欠落するとZIPを受け取った出品者が気づけないため）。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import GarmentCategory


@dataclass(frozen=True)
class StandardPoint:
    label: str
    description: str
    required: bool  # 出品時に必須級とされる項目か
    phase1_supported: bool  # 現在のヒューリスティックで推定できるか


# トップス / アウター / ワンピース共通の上半身項目
_TOP_LIKE_POINTS: list[StandardPoint] = [
    StandardPoint("着丈", "肩の縫い目（衿ぐり後中心）から裾までの長さ", required=True, phase1_supported=True),
    StandardPoint("身幅", "脇下から脇下までの水平幅（=バスト目安 x2ではなく片身頃幅）", required=True, phase1_supported=True),
    StandardPoint("肩幅", "肩の縫い目から縫い目までの直線距離", required=True, phase1_supported=True),
    StandardPoint("袖丈", "肩の縫い目から袖口までの長さ", required=True, phase1_supported=True),
    StandardPoint("裄丈", "背中心から肩を通り袖口までの長さ（肩幅/2 + 袖丈で近似）", required=False, phase1_supported=True),
    StandardPoint("袖口幅", "袖口の開口部の幅", required=False, phase1_supported=False),
    StandardPoint("襟ぐり幅", "衿ぐり（首まわり開口部）の幅", required=False, phase1_supported=False),
]

_BOTTOM_POINTS: list[StandardPoint] = [
    StandardPoint("総丈", "ウエスト上端から裾までの長さ", required=True, phase1_supported=True),
    StandardPoint("ウエスト幅", "ウエスト部分を平置きで測った幅（実周囲は概ねこの2倍）", required=True, phase1_supported=True),
    StandardPoint("股上", "ウエスト上端から股（クロッチポイント）までの長さ", required=True, phase1_supported=False),
    StandardPoint("股下", "股（クロッチポイント）から裾までの長さ", required=True, phase1_supported=False),
    StandardPoint("わたり幅", "股下直下（もも付け根）の水平幅", required=False, phase1_supported=False),
    StandardPoint("裾幅", "裾の開口部の水平幅", required=False, phase1_supported=True),
]

STANDARD_POINTS: dict[GarmentCategory, list[StandardPoint]] = {
    GarmentCategory.TOPS: _TOP_LIKE_POINTS,
    GarmentCategory.OUTER: _TOP_LIKE_POINTS,
    GarmentCategory.DRESS: _TOP_LIKE_POINTS,
    GarmentCategory.BOTTOMS: _BOTTOM_POINTS,
    GarmentCategory.OTHER: [],
}


def standard_points_for(category: GarmentCategory) -> list[StandardPoint]:
    return STANDARD_POINTS.get(category, [])
