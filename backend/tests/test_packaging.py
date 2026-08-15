import zipfile
from pathlib import Path

from app.models.schemas import GarmentCategory, ProductInput, ProductResult
from app.services.packaging import PackagingService


def test_build_zip_contains_expected_entries(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    result = ProductResult(
        product_id="test123",
        input=ProductInput(
            product_name="テストシャツ",
            category=GarmentCategory.TOPS,
            reference_measurement_cm=70.0,
        ),
        flat_lay_images=[],
        output_dir=str(tmp_path),
    )

    zip_path = tmp_path / "test123.zip"
    PackagingService().build_zip(result, images_dir, zip_path)

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert "measurements.json" in names
        assert "condition.md" in names
        assert "copy.md" in names
        assert "summary.json" in names
