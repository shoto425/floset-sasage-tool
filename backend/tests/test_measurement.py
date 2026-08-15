import numpy as np
import pytest

from app.models.schemas import GarmentCategory
from app.services.measurement import MeasurementService

_HEIGHT, _WIDTH = 200, 160
# 肩(狭い) -> 袖(広い) -> 身頃(中間) という3段構成にして、
# 肩幅の算出位置と袖丈の算出位置(=袖の最も張り出した点)を意図的にずらす。
_SHOULDER_Y0, _SHOULDER_Y1 = 10, 38
_SHOULDER_X0, _SHOULDER_X1 = 35, 125
_SLEEVE_Y0, _SLEEVE_Y1 = 38, 73
_SLEEVE_X0, _SLEEVE_X1 = 10, 150
_BODY_Y1 = _HEIGHT - 10
_BODY_X0, _BODY_X1 = 40, 120


def _make_tshirt_mask() -> np.ndarray:
    """肩幅・袖丈の両方が幾何的に区別できる簡易Tシャツ形状のマスクを生成する。"""
    mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    mask[_SHOULDER_Y0:_SHOULDER_Y1, _SHOULDER_X0:_SHOULDER_X1] = 255  # 肩
    mask[_SLEEVE_Y0:_SLEEVE_Y1, _SLEEVE_X0:_SLEEVE_X1] = 255  # 袖（肩より張り出す）
    mask[_SLEEVE_Y1:_BODY_Y1, _BODY_X0:_BODY_X1] = 255  # 身頃
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

    # 袖が肩より張り出しているため袖丈は非ゼロで算出され、
    # 裄丈(肩幅/2+袖丈)もあわせて算出される。袖口幅・襟ぐり幅は非対応として明示される。
    assert labels["袖丈"].value_cm is not None and labels["袖丈"].value_cm > 0
    assert labels["裄丈"].value_cm is not None
    assert labels["袖口幅"].value_cm is None
    assert labels["袖口幅"].method == "not_supported_phase1"
    assert labels["襟ぐり幅"].method == "not_supported_phase1"


def test_estimate_bottoms_includes_unsupported_standard_points():
    mask = np.zeros((200, 100), dtype=np.uint8)
    mask[10:190, 20:80] = 255  # 単純な長方形（パンツを模した平置き形状）

    service = MeasurementService()
    result = service.estimate(
        mask=mask,
        category=GarmentCategory.BOTTOMS,
        reference_label="総丈",
        reference_value_cm=100.0,
    )

    labels = {p.label: p for p in result.points}
    assert labels["総丈"].value_cm == pytest.approx(100.0, rel=0.02)
    assert labels["ウエスト幅"].value_cm is not None
    assert labels["裾幅"].value_cm is not None

    # 股上・股下・わたり幅はPhase1では非対応として明示される
    for label in ("股上", "股下", "わたり幅"):
        assert labels[label].value_cm is None
        assert labels[label].method == "not_supported_phase1"


def test_estimate_raises_on_empty_mask():
    import pytest

    mask = np.zeros((100, 100), dtype=np.uint8)
    service = MeasurementService()
    with pytest.raises(ValueError):
        service.estimate(mask, GarmentCategory.TOPS, "着丈", 70.0)
