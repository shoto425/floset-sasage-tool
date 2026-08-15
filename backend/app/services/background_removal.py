"""背景除去サービス。

`rembg` (U2Net) を第一候補とし、モデルダウンロードに失敗する/未インストールの
オフライン環境では OpenCV GrabCut ベースの簡易セグメンテーションにフォールバックする。
どちらの実装も「BGR画像 -> (RGBA画像, マスク)」という同じインターフェースを満たす。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from app.core.config import settings


class BackgroundRemover(ABC):
    @abstractmethod
    def remove(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BGR画像を受け取り、(RGBA画像, 0/255の2値マスク) を返す。"""


class RembgBackgroundRemover(BackgroundRemover):
    """rembg (U2Net) による背景除去。初回呼び出し時にモデルを遅延ロードする。"""

    def __init__(self) -> None:
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            from rembg import new_session

            self._session = new_session("u2net")
        return self._session

    def remove(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from rembg import remove

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgba = remove(rgb, session=self._ensure_session())
        mask = (rgba[:, :, 3] > 10).astype(np.uint8) * 255
        return rgba, mask


class GrabCutBackgroundRemover(BackgroundRemover):
    """rembg が使えない環境向けのフォールバック。

    平置き撮影は背景が単色に近いことを前提に、画像端を背景シードとした
    GrabCut で前景（衣類）マスクを推定する。rembg より精度は落ちる。
    """

    def remove(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_bgr.shape[:2]
        mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        margin_x, margin_y = int(w * 0.12), int(h * 0.12)
        mask[margin_y : h - margin_y, margin_x : w - margin_x] = cv2.GC_PR_FGD

        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(image_bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

        binary_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        # ノイズ除去
        binary_mask = cv2.morphologyEx(
            binary_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
        )

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgba = np.dstack([rgb, binary_mask])
        return rgba, binary_mask


def get_background_remover() -> BackgroundRemover:
    if settings.background_removal_backend == "rembg":
        try:
            import rembg  # noqa: F401

            return RembgBackgroundRemover()
        except ImportError:
            pass
    return GrabCutBackgroundRemover()
