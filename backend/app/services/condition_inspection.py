"""傷・状態検品サービス。

## Phase 1: ヒューリスティック実装 (`HeuristicConditionInspector`)
衣類マスクの領域をグリッドに分割し、セルごとに
- エッジ密度（Canny検出後の白画素率）→ 局所的に高いと「ほつれ・穴・破れ」の可能性
- 明度（HSVのV平均）→ 周囲より局所的に明るいと「色落ち・色あせ」の可能性
を計算し、全セル平均からのZスコアが閾値を超えたセルを異常候補として報告する。
あくまで統計的な異常検知であり、誤検知（生地の柄・シワを傷と誤認する等）は
発生しうる前提で、出品者による最終確認を促す文言をセットにする。

## Phase 2 以降
`LLMConditionInspector` として Claude Vision に画像を渡し、自然文で状態を
説明させる実装に拡張する（`llm_client.py` 経由）。インターフェースは共通。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from app.core.config import settings
from app.models.schemas import ConditionIssue, ConditionSeverity


class ConditionInspector(ABC):
    @abstractmethod
    def inspect(self, image_bgr: np.ndarray, mask: np.ndarray) -> list[ConditionIssue]:
        ...


def _location_hint(row: int, col: int, n_rows: int, n_cols: int) -> str:
    vertical = "上" if row < n_rows / 3 else ("下" if row >= 2 * n_rows / 3 else "中央")
    horizontal = "左" if col < n_cols / 3 else ("右" if col >= 2 * n_cols / 3 else "中央")
    if vertical == "中央" and horizontal == "中央":
        return "中央付近"
    if vertical == "中央":
        return f"{horizontal}側"
    if horizontal == "中央":
        return f"{vertical}部"
    return f"{vertical}{horizontal}"


class HeuristicConditionInspector(ConditionInspector):
    def __init__(self, n_rows: int = 6, n_cols: int = 6, zscore_threshold: float | None = None) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.zscore_threshold = (
            zscore_threshold
            if zscore_threshold is not None
            else settings.condition_edge_density_zscore_threshold
        )

    def inspect(self, image_bgr: np.ndarray, mask: np.ndarray) -> list[ConditionIssue]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return []
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()

        edges = cv2.Canny(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), 60, 150)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]

        cell_h = max(1, (y1 - y0) // self.n_rows)
        cell_w = max(1, (x1 - x0) // self.n_cols)

        edge_densities: list[tuple[int, int, float, tuple[int, int, int, int]]] = []
        brightness_values: list[tuple[int, int, float, tuple[int, int, int, int]]] = []

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                cy0, cy1 = y0 + row * cell_h, min(y1, y0 + (row + 1) * cell_h)
                cx0, cx1 = x0 + col * cell_w, min(x1, x0 + (col + 1) * cell_w)
                cell_mask = mask[cy0:cy1, cx0:cx1]
                if cell_mask.size == 0 or cell_mask.sum() == 0:
                    continue
                cell_edges = edges[cy0:cy1, cx0:cx1]
                fg_ratio = cell_mask.mean() / 255
                if fg_ratio < 0.3:
                    continue  # 衣類がほとんど写っていないセルは無視
                edge_density = float((cell_edges > 0).mean())
                brightness = float(value_channel[cy0:cy1, cx0:cx1][cell_mask > 0].mean())
                bbox = (int(cx0), int(cy0), int(cx1 - cx0), int(cy1 - cy0))
                edge_densities.append((row, col, edge_density, bbox))
                brightness_values.append((row, col, brightness, bbox))

        issues: list[ConditionIssue] = []
        issues.extend(self._flag_anomalies(
            edge_densities,
            description="ほつれ・傷の可能性",
            unit_name="edge",
        ))
        issues.extend(self._flag_anomalies(
            brightness_values,
            description="色落ち・色あせの可能性",
            unit_name="brightness",
            direction="high",
        ))
        return issues

    def _flag_anomalies(
        self,
        values: list[tuple[int, int, float, tuple[int, int, int, int]]],
        description: str,
        unit_name: str,
        direction: str = "high",
    ) -> list[ConditionIssue]:
        if len(values) < 3:
            return []
        arr = np.array([v[2] for v in values])
        mean, std = arr.mean(), arr.std()
        if std < 1e-6:
            return []

        issues: list[ConditionIssue] = []
        for row, col, value, bbox in values:
            z = (value - mean) / std
            score = z if direction == "high" else -z
            if score < self.zscore_threshold:
                continue
            severity = (
                ConditionSeverity.HIGH
                if score > self.zscore_threshold + 1
                else ConditionSeverity.MEDIUM
            )
            issues.append(
                ConditionIssue(
                    description=f"{description}（{unit_name}異常スコア {score:.1f}）",
                    severity=severity,
                    location_hint=_location_hint(row, col, self.n_rows, self.n_cols),
                    recommend_closeup=True,
                    bbox_px=bbox,
                    detected_by="heuristic_cv",
                )
            )
        return issues


def get_condition_inspector() -> ConditionInspector:
    if settings.condition_backend == "llm_vision":
        from app.services.condition_inspection_llm import LLMConditionInspector

        return LLMConditionInspector()
    return HeuristicConditionInspector()
