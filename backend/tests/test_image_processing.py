from pathlib import Path

import numpy as np

from app.services.image_processing import ImageProcessingService


def test_to_white_background_jpeg_produces_canvas(tmp_path: Path):
    h, w = 80, 60
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 200  # R
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:70, 10:50] = 255  # 衣類領域

    service = ImageProcessingService(canvas_size=400, padding_ratio=0.1)
    out_path = tmp_path / "out.jpg"
    width, height = service.to_white_background_jpeg(rgba, mask, out_path)

    assert out_path.exists()
    assert width == 400
    assert height == 400
