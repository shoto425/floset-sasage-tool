"""採寸サービス（Phase 1: ヒューリスティック実装）。

## 手法の概要
カメラキャリブレーションや基準物（メジャー等）を使わずに実寸を出すため、
Phase 1 では「出品者が1箇所だけ実測してくれた値（例: 着丈）」を基準として、
マスク形状のピクセル比率から他の採寸項目を推定する。

    scale_cm_per_px = reference_value_cm / (基準項目に対応するマスク上のpx長)

## 既知の限界（必ずドキュメント上でも明示すること）
- 衣類が完全な平置き・正面から撮影されている前提（斜め撮影だと誤差が拡大する）
- 肩幅・袖丈は「マスクの幅プロファイル」から幾何的に近似しているだけで、
  縫い目や生地のたわみは考慮していない
- Phase 2 で MediaPipe / SAM のキーポイント検出に置き換えて精度を上げる
"""
from __future__ import annotations

import cv2
import numpy as np

from app.models.schemas import GarmentCategory, MeasurementPoint, MeasurementSet


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("マスクから輪郭を検出できませんでした")
    return max(contours, key=cv2.contourArea)


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

        # 基準項目自体も結果に含める（着丈など）
        points.insert(
            0,
            MeasurementPoint(
                label=reference_label,
                value_cm=round(reference_value_cm, 1),
                confidence=1.0,
                method="user_reference",
            ),
        )

        return MeasurementSet(
            category=category,
            reference_label=reference_label,
            reference_value_cm=reference_value_cm,
            scale_cm_per_px=scale,
            points=points,
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

        return [
            MeasurementPoint(
                label="肩幅",
                value_cm=round(shoulder_width_px * scale, 1),
                confidence=0.55,
            ),
            MeasurementPoint(
                label="身幅",
                value_cm=round(body_width_px * scale, 1),
                confidence=0.6,
            ),
            MeasurementPoint(
                label="袖丈",
                value_cm=round(sleeve_length_px * scale, 1) if sleeve_length_px else None,
                confidence=0.4 if sleeve_length_px else 0.0,
            ),
        ]

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

        hip_y = top_y + int(total_length_px * 0.2)
        hip_y = min(profile.keys(), key=lambda y: abs(y - hip_y))
        hip_x0, hip_x1 = profile[hip_y]

        return [
            MeasurementPoint(
                label="ウエスト幅",
                value_cm=round((waist_x1 - waist_x0) * scale, 1),
                confidence=0.5,
            ),
            MeasurementPoint(
                label="股上〜ヒップ幅",
                value_cm=round((hip_x1 - hip_x0) * scale, 1),
                confidence=0.45,
            ),
        ]
