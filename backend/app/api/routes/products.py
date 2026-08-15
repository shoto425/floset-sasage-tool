"""商品処理API。

Phase 1 は同期処理（アップロード→即結果）。動画が長時間・高解像度になる
Phase 2 ではジョブキュー化する想定だが、ルーティングI/Fは変えずに済むよう
リクエスト/レスポンスの形は将来の非同期化を見込んで product_id を軸にしてある。
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.pipeline import pipeline
from app.models.schemas import GarmentCategory, ProductInput, ProductResult
from app.storage.file_store import file_store

router = APIRouter(prefix="/products", tags=["products"])

_results: dict[str, ProductResult] = {}


@router.post("/process", response_model=ProductResult)
async def process_video(
    video: UploadFile = File(..., description="1本の動画ファイル"),
    product_name: str = Form(...),
    category: GarmentCategory = Form(GarmentCategory.TOPS),
    brand: str | None = Form(None),
    reference_measurement_label: str = Form("着丈"),
    reference_measurement_cm: float = Form(...),
    notes: str | None = Form(None),
) -> ProductResult:
    product_input = ProductInput(
        product_name=product_name,
        category=category,
        brand=brand,
        reference_measurement_label=reference_measurement_label,
        reference_measurement_cm=reference_measurement_cm,
        notes=notes,
    )

    tmp_id = file_store.new_product_id()
    video_path = file_store.upload_path(tmp_id, video.filename or "upload.mp4")
    content = await video.read()
    video_path.write_bytes(content)

    try:
        result = pipeline.process_video(str(video_path), product_input)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    _results[result.product_id] = result
    return result


@router.get("/{product_id}", response_model=ProductResult)
async def get_product(product_id: str) -> ProductResult:
    result = _results.get(product_id)
    if result is None:
        raise HTTPException(status_code=404, detail="指定された product_id が見つかりません")
    return result


@router.get("/{product_id}/zip")
async def download_zip(product_id: str) -> FileResponse:
    result = _results.get(product_id)
    if result is None or not result.zip_path:
        raise HTTPException(status_code=404, detail="ZIPが見つかりません")
    return FileResponse(
        result.zip_path,
        media_type="application/zip",
        filename=f"{product_id}.zip",
    )


@router.get("/{product_id}/images/{file_name}")
async def download_image(product_id: str, file_name: str) -> FileResponse:
    result = _results.get(product_id)
    if result is None:
        raise HTTPException(status_code=404, detail="指定された product_id が見つかりません")
    if not any(img.file_name == file_name for img in result.flat_lay_images):
        raise HTTPException(status_code=404, detail="指定された画像が見つかりません")

    image_path = file_store.images_dir(product_id) / file_name
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="指定された画像が見つかりません")
    return FileResponse(image_path, media_type="image/jpeg")
