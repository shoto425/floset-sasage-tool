"""商品ごとの納品ZIP生成サービス。

ZIP構成:
    images/*.jpg           - 平置きJPEG群
    measurements.json      - 採寸データ
    condition.md           - 状態検品レポート
    copy.md                - Instagram原稿 / EC原稿
    summary.json           - ProductResult 全体（機械可読な統合データ）
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from app.models.schemas import ProductResult


class PackagingService:
    def build_zip(self, result: ProductResult, images_dir: Path, zip_path: Path) -> Path:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for image in result.flat_lay_images:
                image_path = images_dir / image.file_name
                if image_path.exists():
                    zf.write(image_path, arcname=f"images/{image.file_name}")

            zf.writestr("measurements.json", self._measurements_json(result))
            zf.writestr("condition.md", self._condition_markdown(result))
            zf.writestr("copy.md", self._copy_markdown(result))
            zf.writestr("summary.json", result.model_dump_json(indent=2))
        return zip_path

    def _measurements_json(self, result: ProductResult) -> str:
        if result.measurements is None:
            return "{}"
        return result.measurements.model_dump_json(indent=2)

    def _condition_markdown(self, result: ProductResult) -> str:
        lines = [f"# 状態検品レポート — {result.input.product_name}", ""]
        if not result.condition_issues:
            lines.append("目立った傷・状態の問題は検出されませんでした。")
        for issue in result.condition_issues:
            lines.append(f"## {issue.location_hint}")
            lines.append(f"- 内容: {issue.description}")
            lines.append(f"- 深刻度: {issue.severity.value}")
            lines.append(
                f"- 接写推奨: {'あり' if issue.recommend_closeup else 'なし'}"
            )
            lines.append(f"- 検出方法: {issue.detected_by}")
            lines.append("")
        return "\n".join(lines)

    def _copy_markdown(self, result: ProductResult) -> str:
        if result.generated_copy is None:
            return "原稿は生成されませんでした。"
        hashtags = " ".join(result.generated_copy.hashtags)
        return (
            f"# Instagram原稿\n\n{result.generated_copy.instagram_caption}\n\n{hashtags}\n\n"
            f"# EC原稿\n\n{result.generated_copy.ec_description}\n"
        )
