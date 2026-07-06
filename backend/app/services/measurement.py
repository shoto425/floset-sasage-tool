"""採寸サービス（Phase 1: ヒューリスティック実装）。

## 手法の概要
カメラキャリブレーションや基準物（メジャー等）を使わずに実寸を出すため、
Phase 1 では「出品者が1箇所だけ実測してくれた値（例: 着丈）」を基準として、
マスク形状のピクセル比率から他の採寸項目を推定する。

    scale_cm_per_px = reference_value_cm / (基準項目に対応するマスク上のpx長)

標準採寸項目そのものの定義は `measurement_standards.py` を参照。本サービスは
そこで定義された項目のうち Phase1 で計算可能なものを算出し、非対応の項目は
`value_cm=None, method="not_supported_phase1"` として結果に含めることで、
「項目自体は認識しているが未対応」であることを出力上で明示する。

## 既知の限界（必ずドキュメント上でも明示すること）
- 衣類が完全な平置き・正面から撮影されている前提（斜め撮影だと誤差が拡大する）
- 肩幅・袖丈は「マスクの幅プロファイル」から幾何的に近似しているだけで、
  縫い目や生地のたわみは考慮していない
- ボトムスの股上・股下・わたり幅はクロッチポイント（股の分岐点）の検出が
  必要でありPhase1のマスク形状だけでは信頼できる推定ができないため非対応
- Phase 2 で MediaPipe / SAM のキーポイント検出に置き換えて精度を上げる
  （詳細は `docs/MEASUREMENT_STANDARDS.md` の改善計画を参照）
"""
from __future__ import annotations

import cv2
import numpy as np

from app.models.schemas import GarmentCategory, MeasurementPoint, MeasurementSet
from app.services.measurement_standards import standard_points_for


def _row_width_profile(mask: np.ndarray) -> dict[int, tuple[int, int]]:
    """各y座標について、マスクが存在する x の最小・最大値を返す。"""
    profile: dict[int, tuple[int, int]] = {}
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return profile
    for y in np.unique(ys):
        row_xs = xs[ys == y]
        profile[int(y)] = (int(row_xs.min()), int(row_xs.max()))
    return profile


class MeasurementService:
    """マスク画像 + 基準実測値から MeasurementSet を組み立てる。"""

    def estimate(
        self,
        mask: np.ndarray,
        category: GarmentCategory,
        reference_label: str,
        reference_value_cm: float,
    ) -> MeasurementSet:
        profile = _row_width_profile(mask)
        if not profile:
            raise ValueError("マスクが空のため採寸できません")

        ys = sorted(profile.keys())
        top_y, bottom_y = ys[0], ys[-1]
        total_length_px = bottom_y - top_y
        if total_length_px <= 0:
            raise ValueError("マスクの縦方向の長さが不正です")

        scale = reference_value_cm / total_length_px

        if category in (GarmentCategory.TOPS, GarmentCategory.OUTER, GarmentCategory.DRESS):
            points = self._estimate_top_like(profile, top_y, bottom_y, scale)
        else:
            points = self._estimate_bottoms(profile, top_y, bottom_y, scale)

        # 基準項目自体も結果に含める（着丈など）。既に算出済みなら上書きしない。
        computed_labels = {p.label for p in points}
        if reference_label not in computed_labels:
            points.insert(
                0,
                MeasurementPoint(
                    label=reference_label,
                    value_cm=round(reference_value_cm, 1),
                    confidence=1.0,
                    method="user_reference",
                ),
            )

        self._fill_unsupported_standard_points(points, category)

        return MeasurementSet(
            category=category,
            reference_label=reference_label,
            reference_value_cm=reference_value_cm,
            scale_cm_per_px=scale,
            points=points,
        )

    def _fill_unsupported_standard_points(
        self, points: list[MeasurementPoint], category: GarmentCategory
    ) -> None:
        """標準項目のうちまだ算出されていないものを、非対応として明示的に追加する。"""
        existing_labels = {p.label for p in points}
        for standard in standard_points_for(category):
            if standard.label in existing_labels:
                continue
            points.append(
                MeasurementPoint(
                    label=standard.label,
                    value_cm=None,
                    confidence=0.0,
                    method="not_supported_phase1",
                )
            )

    def _estimate_top_like(
        self,
        profile: dict[int, tuple[int, int]],
        top_y: int,
        bottom_y: int,
        scale: float,
    ) -> list[MeasurementPoint]:
        total_length_px = bottom_y - top_y
        shoulder_band = [
            y for y in profile if y <= top_y + int(total_length_px * 0.15)
        ]
        shoulder_y = max(
            shoulder_band, key=lambda y: profile[y][1] - profile[y][0]
        )
        shoulder_x0, shoulder_x1 = profile[shoulder_y]
        shoulder_width_px = shoulder_x1 - shoulder_x0

        body_y = top_y + int(total_length_px * 0.45)
        body_y = min(profile.keys(), key=lambda y: abs(y - body_y))
        body_x0, body_x1 = profile[body_y]
        body_width_px = body_x1 - body_x0

        sleeve_band_end = top_y + int(total_length_px * 0.4)
        sleeve_band = {y: v for y, v in profile.items() if shoulder_y <= y <= sleeve_band_end}
        sleeve_length_px = 0.0
        if sleeve_band:
            left_tip_y = min(sleeve_band, key=lambda y: sleeve_band[y][0])
            left_tip_x = sleeve_band[left_tip_y][0]
            right_tip_y = max(sleeve_band, key=lambda y: sleeve_band[y][1])
            right_tip_x = sleeve_band[right_tip_y][1]

            left_len = float(
                np.hypot(shoulder_x0 - left_tip_x, shoulder_y - left_tip_y)
            )
            right_len = float(
                np.hypot(shoulder_x1 - right_tip_x, shoulder_y - right_tip_y)
            )
            sleeve_length_px = (left_len + right_len) / 2

        shoulder_width_cm = round(shoulder_width_px * scale, 1)
        sleeve_length_cm = round(sleeve_length_px * scale, 1) if sleeve_length_px else None

        points = [
            MeasurementPoint(
                label="肩幅",
                value_cm=shoulder_width_cm,
                confidence=0.55,
            ),
            MeasurementPoint(
                label="身幅",
                value_cm=round(body_width_px * scale, 1),
                confidence=0.6,
            ),
            MeasurementPoint(
                label="袖丈",
                value_cm=sleeve_length_cm,
                confidence=0.4 if sleeve_length_px else 0.0,
            ),
        ]

        if sleeve_length_cm is not None:
            # 裄丈 = 肩幅の半分 + 袖丈（和裁の慣習的な近似式）
            yuki_cm = round(shoulder_width_cm / 2 + sleeve_length_cm, 1)
            points.append(
                MeasurementPoint(label="裄丈", value_cm=yuki_cm, confidence=0.35, method="derived_formula")
            )

        return points

    def _estimate_bottoms(
        self,
        profile: dict[int, tuple[int, int]],
        top_y: int,
        bottom_y: int,
        scale: float,
    ) -> list[MeasurementPoint]:
        total_length_px = bottom_y - top_y

        waist_y = top_y + int(total_length_px * 0.05)
        waist_y = min(profile.keys(), key=lambda y: abs(y - waist_y))
        waist_x0, waist_x1 = profile[waist_y]

        hem_y = bottom_y - int(total_length_px * 0.03)
        hem_y = min(profile.keys(), key=lambda y: abs(y - hem_y))
        hem_x0, hem_x1 = profile[hem_y]

        return [
            MeasurementPoint(
                label="総丈",
                value_cm=round(total_length_px * scale, 1),
                confidence=0.6,
            ),
            MeasurementPoint(
                label="ウエスト幅",
                value_cm=round((waist_x1 - waist_x0) * scale, 1),
                confidence=0.5,
            ),
            MeasurementPoint(
                label="裾幅",
                value_cm=round((hem_x1 - hem_x0) * scale, 1),
                confidence=0.5,
            ),
        ]
