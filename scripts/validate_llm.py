"""Claude API連携（原稿生成 / Vision状態検品）の手動検証スクリプト。

pytestのCIには含めない（実APIコールが発生し課金・レイテンシがあるため）。
`SASAGE_ANTHROPIC_API_KEY` を設定した上で手動実行し、結果を目視確認する。

    SASAGE_ANTHROPIC_API_KEY=... SASAGE_LLM_BACKEND=claude \
        python scripts/validate_llm.py

実店舗の古着動画がまだ無いため、本スクリプトは「合成画像（意図的に傷を
描き込んだダミー平置き画像）」を使う。あくまで Claude API 連携・プロンプト
設計そのものが機能しているかを確認するための代用であり、実物のシワ・素材感・
色合いに対する精度検証ではない点に注意（詳細は docs/PROGRESS.md）。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.config import settings  # noqa: E402
from app.models.schemas import GarmentCategory, ProductInput  # noqa: E402
from app.services.condition_inspection_llm import LLMConditionInspector  # noqa: E402
from app.services.copywriting import CopywritingService  # noqa: E402


@dataclass
class SyntheticProduct:
    name: str
    category: GarmentCategory
    color_bgr: tuple[int, int, int]
    defect: str | None  # 期待される検出内容の説明（Noneなら傷なしcontrol）
    defect_keywords: tuple[str, ...]  # descriptionに含まれることを期待するキーワード
    location_keywords: tuple[str, ...]  # location_hintに含まれることを期待するキーワード


def _make_garment_image(product: SyntheticProduct) -> tuple[np.ndarray, np.ndarray]:
    h, w = 500, 400
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    x0, y0, x1, y1 = 60, 40, 340, 460
    cv2.rectangle(img, (x0, y0), (x1, y1), product.color_bgr, -1)
    # 袖
    cv2.rectangle(img, (10, 40), (60, 160), product.color_bgr, -1)
    cv2.rectangle(img, (340, 40), (390, 160), product.color_bgr, -1)

    if product.defect == "hole_sleeve":
        cv2.circle(img, (35, 100), 12, (255, 255, 255), -1)
        cv2.circle(img, (35, 100), 12, (30, 30, 30), 2)
    elif product.defect == "fraying_hem":
        for i in range(x0, x1, 8):
            cv2.line(img, (i, y1), (i + 4, y1 + 14), (30, 30, 30), 2)
    elif product.defect == "fading_patch":
        light = tuple(min(255, c + 90) for c in product.color_bgr)
        cv2.ellipse(img, (200, 150), (60, 40), 0, 0, 360, light, -1)
    elif product.defect == "stain":
        cv2.circle(img, (220, 300), 30, (60, 90, 130), -1)
    elif product.defect == "tear_front":
        pts = np.array([[180, 250], [230, 250], [205, 310]])
        cv2.fillPoly(img, [pts], (255, 255, 255))

    mask = np.any(img != 255, axis=2).astype(np.uint8) * 255
    return img, mask


PRODUCTS: list[SyntheticProduct] = [
    SyntheticProduct("無地Tシャツ(良好・control)", GarmentCategory.TOPS, (120, 120, 200), None, (), ()),
    SyntheticProduct("Tシャツ(左袖に穴)", GarmentCategory.TOPS, (120, 160, 90), "hole_sleeve", ("穴", "虫食い", "破れ", "傷"), ("袖",)),
    SyntheticProduct("シャツ(裾にほつれ)", GarmentCategory.TOPS, (200, 180, 140), "fraying_hem", ("ほつれ", "擦り切れ", "傷み"), ("裾", "下")),
    SyntheticProduct("デニムパンツ(色落ちパッチ)", GarmentCategory.BOTTOMS, (150, 90, 40), "fading_patch", ("色落ち", "色あせ", "退色"), ("中央", "中")),
    SyntheticProduct("スカート(シミ)", GarmentCategory.BOTTOMS, (130, 130, 130), "stain", ("シミ", "汚れ", "変色"), ("下", "裾")),
    SyntheticProduct("ジャケット(前面破れ)", GarmentCategory.OUTER, (80, 80, 80), "tear_front", ("破れ", "穴", "裂け"), ("中央", "中")),
    SyntheticProduct("ニット(良好・control2)", GarmentCategory.TOPS, (90, 140, 200), None, (), ()),
]


def _keyword_hit(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


def main() -> None:
    if not settings.anthropic_api_key:
        print("SASAGE_ANTHROPIC_API_KEY が未設定です。処理を中断します。")
        sys.exit(1)

    inspector = LLMConditionInspector()
    copy_service = CopywritingService()

    results = []
    for product in PRODUCTS:
        image_bgr, mask = _make_garment_image(product)
        issues = inspector.inspect(image_bgr, mask)

        detected = False
        matched_issue = None
        if product.defect is not None:
            for issue in issues:
                if _keyword_hit(issue.description, product.defect_keywords) and (
                    not product.location_keywords
                    or _keyword_hit(issue.location_hint, product.location_keywords)
                ):
                    detected = True
                    matched_issue = issue
                    break
        else:
            detected = len(issues) == 0  # controlは「検出なし」が正解

        product_input = ProductInput(
            product_name=product.name,
            category=product.category,
            reference_measurement_cm=70.0,
        )
        copy = copy_service.generate(product_input, None, issues)

        results.append(
            {
                "product": product.name,
                "expected_defect": product.defect,
                "raw_issues": [i.model_dump() for i in issues],
                "vision_correct": detected,
                "matched_issue": matched_issue.model_dump() if matched_issue else None,
                "copy_generated_by": copy.generated_by,
                "copy": copy.model_dump(),
            }
        )
        print(f"[{'OK' if detected else 'NG'}] {product.name}: issues={len(issues)} copy_by={copy.generated_by}")

    out_dir = Path(os.environ.get("SASAGE_VALIDATION_OUT_DIR", "/tmp"))
    out_path = out_dir / "sasage_llm_validation_raw.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    vision_correct = sum(1 for r in results if r["vision_correct"])
    copy_llm = sum(1 for r in results if r["copy_generated_by"] == "llm")
    print()
    print(f"Vision検品: {vision_correct}/{len(results)} 件が期待通り")
    print(f"原稿生成(LLMバックエンド): {copy_llm}/{len(results)} 件")
    print(f"詳細な生データ: {out_path}")


if __name__ == "__main__":
    main()
