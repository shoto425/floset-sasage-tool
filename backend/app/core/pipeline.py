"""Phase 1 パイプラインオーケストレーター。

「動画1本 -> 平置き画像群 + 採寸 + 状態検品 + 原稿 + ZIP」を1回の呼び出しで
完結させる。各ステップは `app.services.*` の独立サービスに委譲しているため、
ここは「呼び出し順序」と「エラー時の握りつぶし方針（1商品全体を失敗させない）」
の責務だけを持つ。
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import settings
from app.models.schemas import FlatLayImage, ProductInput, ProductResult
from app.services.background_removal import get_background_remover
from app.services.condition_inspection import get_condition_inspector
from app.services.copywriting import CopywritingService
from app.services.frame_extraction import FrameExtractionService
from app.services.image_processing import ImageProcessingService
from app.services.measurement import MeasurementService
from app.services.packaging import PackagingService
from app.storage.file_store import file_store

logger = logging.getLogger(__name__)


class SasagePipeline:
    def __init__(self) -> None:
        self.frame_extractor = FrameExtractionService()
        self.background_remover = get_background_remover()
        self.image_processor = ImageProcessingService()
        self.measurement_service = MeasurementService()
        self.condition_inspector = get_condition_inspector()
        self.copywriting_service = CopywritingService()
        self.packaging_service = PackagingService()

    def process_video(self, video_path: str, product_input: ProductInput) -> ProductResult:
        product_id = file_store.new_product_id()
        images_dir = file_store.images_dir(product_id)
        warnings: list[str] = []

        frames = self.frame_extractor.extract(video_path)
        if not frames:
            raise ValueError("動画からフレームを抽出できませんでした")

        flat_lay_images: list[FlatLayImage] = []
        best_mask = None
        best_frame_bgr = None

        for i, frame in enumerate(frames):
            try:
                rgba, mask = self.background_remover.remove(frame.image_bgr)
            except Exception:
                logger.exception("背景除去に失敗しました (frame=%s)", frame.index)
                warnings.append(f"フレーム{frame.index}の背景除去に失敗しました")
                continue

            file_name = f"flatlay_{i + 1:02d}.jpg"
            out_path = images_dir / file_name
            width, height = self.image_processor.to_white_background_jpeg(rgba, mask, out_path)

            flat_lay_images.append(
                FlatLayImage(
                    file_name=file_name,
                    source_frame_index=frame.index,
                    sharpness_score=frame.sharpness,
                    width=width,
                    height=height,
                    is_primary=(i == 0),
                )
            )

            # 採寸・検品には最も鮮明な代表フレーム（先頭 = sharpness降順の1枚目）を使う
            if best_mask is None:
                best_mask = mask
                best_frame_bgr = frame.image_bgr

        if not flat_lay_images or best_mask is None:
            raise ValueError("平置き画像を1枚も生成できませんでした")

        measurements = None
        try:
            measurements = self.measurement_service.estimate(
                best_mask,
                product_input.category,
                product_input.reference_measurement_label,
                product_input.reference_measurement_cm,
            )
        except Exception:
            logger.exception("採寸に失敗しました")
            warnings.append("採寸に失敗しました。手動での採寸を推奨します。")

        condition_issues = []
        try:
            condition_issues = self.condition_inspector.inspect(best_frame_bgr, best_mask)
        except Exception:
            logger.exception("状態検品に失敗しました")
            warnings.append("状態検品に失敗しました。目視確認を推奨します。")

        generated_copy = self.copywriting_service.generate(product_input, measurements, condition_issues)

        result = ProductResult(
            product_id=product_id,
            input=product_input,
            flat_lay_images=flat_lay_images,
            measurements=measurements,
            condition_issues=condition_issues,
            generated_copy=generated_copy,
            output_dir=str(file_store.product_dir(product_id)),
            warnings=warnings,
        )

        zip_path = file_store.zip_path(product_id)
        self.packaging_service.build_zip(result, images_dir, zip_path)
        result.zip_path = str(zip_path)

        return result


pipeline = SasagePipeline()
