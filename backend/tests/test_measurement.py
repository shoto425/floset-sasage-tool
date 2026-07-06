import numpy as np
import pytest

from app.models.schemas import GarmentCategory
from app.services.measurement import MeasurementService

_HEIGHT, _WIDTH = 200, 160
_SHOULDER_Y0, _SHOULDER_Y1 = 10, 30
_SHOULDER_X0, _SHOULDER_X1 = 10, 150
_BODY_Y1 = _HEIGHT - 10
_BODY_X0, _BODY_X1 = 40, 120


def _make_tshirt_mask() -> np.ndarray:
    """肩幅がある簡易Tシャツ形状のマスクを生成する。"""
    mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    mask[_SHOULDER_Y0:_SHOULDER_Y1, _SHOULDER_X0:_SHOULDER_X1] = 255  # 肩〜袖口
    mask[_SHOULDER_Y1:_BODY_Y1, _BODY_X0:_BODY_X1] = 255  # 身頃
    return mask


def test_estimate_top_like_returns_expected_points():
    mask = _make_tshirt_mask()
    service = MeasurementService()
    reference_cm = 76.0

    result = service.estimate(
        mask=mask,
        category=GarmentCategory.TOPS,
        reference_label="着丈",
        reference_value_cm=reference_cm,
    )

    top_y, bottom_y = _SHOULDER_Y0, _BODY_Y1 - 1
    scale = reference_cm / (bottom_y - top_y)

    labels = {p.label: p for p in result.points}
    assert "着丈" in labels
    assert labels["着丈"].value_cm == reference_cm

    expected_shoulder_cm = (_SHOULDER_X1 - 1 - _SHOULDER_X0) * scale
    assert "肩幅" in labels
    assert labels["肩幅"].value_cm == pytest.approx(expected_shoulder_cm, rel=0.02)

    expected_body_cm = (_BODY_X1 - 1 - _BODY_X0) * scale
    assert "身幅" in labels
    assert labels["身幅"].value_cm == pytest.approx(expected_body_cm, rel=0.05)

    assert result.scale_cm_per_px == pytest.approx(scale, rel=1e-6)


def test_estimate_raises_on_empty_mask():
    import pytest

    mask = np.zeros((100, 100), dtype=np.uint8)
    service = MeasurementService()
    with pytest.raises(ValueError):
        service.estimate(mask, GarmentCategory.TOPS, "着丈", 70.0)
