"""背景除去後の画像を、EC掲載向けの白抜き平置き画像に仕上げるサービス。

処理内容:
1. マスクの外接矩形で衣類部分をクロップ
2. 正方形/指定比率のキャンバスに、余白比率を保って中央配置
3. 透過部分を白(255,255,255)で塗りつぶし、JPEGとして書き出し
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings


class ImageProcessingService:
    def __init__(
        self,
        canvas_size: int | None = None,
        padding_ratio: float | None = None,
    ) -> None:
        self.canvas_size = canvas_size or settings.output_canvas_size
        self.padding_ratio = (
            padding_ratio if padding_ratio is not None else settings.output_padding_ratio
        )

    def to_white_background_jpeg(
        self, rgba_image: np.ndarray, mask: np.ndarray, out_path: Path
    ) -> tuple[int, int]:
        cropped_rgba, cropped_mask = self._crop_to_content(rgba_image, mask)
        composed = self._compose_on_white_canvas(cropped_rgba, cropped_mask)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(composed, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return composed.shape[1], composed.shape[0]

    def _crop_to_content(
        self, rgba_image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return rgba_image, mask
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        return rgba_image[y0 : y1 + 1, x0 : x1 + 1], mask[y0 : y1 + 1, x0 : x1 + 1]

    def _compose_on_white_canvas(self, rgba_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        content_h, content_w = rgba_image.shape[:2]
        usable = self.canvas_size * (1 - 2 * self.padding_ratio)
        scale = usable / max(content_h, content_w)
        new_w, new_h = max(1, int(content_w * scale)), max(1, int(content_h * scale))

        resized_rgb = cv2.resize(rgba_image[:, :, :3], (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((self.canvas_size, self.canvas_size, 3), 255, dtype=np.uint8)
        off_x = (self.canvas_size - new_w) // 2
        off_y = (self.canvas_size - new_h) // 2

        alpha = (resized_mask.astype(np.float32) / 255.0)[:, :, None]
        roi = canvas[off_y : off_y + new_h, off_x : off_x + new_w].astype(np.float32)
        blended = roi * (1 - alpha) + resized_rgb.astype(np.float32) * alpha
        canvas[off_y : off_y + new_h, off_x : off_x + new_w] = blended.astype(np.uint8)
        return canvas
