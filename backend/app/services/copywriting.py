"""Instagram / EC 向け原稿生成サービス。

Claude API が使えればそれを使い、使えない場合（APIキー未設定・呼び出し失敗）は
採寸・状態検品の結果から機械的にテンプレート文を組み立てるフォールバックへ
自動的に切り替える。呼び出し元はどちらのバックエンドかを意識しなくてよい。
"""
from __future__ import annotations

from app.core.config import settings
from app.models.schemas import ConditionIssue, GarmentCategory, GeneratedCopy, MeasurementSet, ProductInput
from app.services.llm_client import claude_client

_SYSTEM_PROMPT = (
    "あなたは古着・ヴィンテージ古着ECショップの敏腕コピーライターです。"
    "採寸データと検品情報をもとに、購入者が安心して判断できる正確さと、"
    "商品の魅力が伝わる言葉選びを両立した原稿を作成してください。"
)


def _condition_summary_text(issues: list[ConditionIssue]) -> str:
    if not issues:
        return "目立った傷や汚れは検出されていません（AI検品の結果。念のため実物のご確認を推奨します）。"
    lines = [f"- {i.location_hint}: {i.description}" for i in issues]
    return "\n".join(lines)


def _measurement_summary_text(measurements: MeasurementSet | None) -> str:
    if measurements is None:
        return "採寸データなし"
    return "\n".join(
        f"- {p.label}: {p.value_cm}cm" if p.value_cm is not None else f"- {p.label}: 計測不可"
        for p in measurements.points
    )


class CopywritingService:
    def generate(
        self,
        product_input: ProductInput,
        measurements: MeasurementSet | None,
        condition_issues: list[ConditionIssue],
    ) -> GeneratedCopy:
        if settings.llm_backend == "claude":
            llm_copy = self._generate_with_llm(product_input, measurements, condition_issues)
            if llm_copy is not None:
                return llm_copy
        return self._generate_fallback(product_input, measurements, condition_issues)

    def _generate_with_llm(
        self,
        product_input: ProductInput,
        measurements: MeasurementSet | None,
        condition_issues: list[ConditionIssue],
    ) -> GeneratedCopy | None:
        user_prompt = (
            f"商品名: {product_input.product_name}\n"
            f"ブランド: {product_input.brand or '不明'}\n"
            f"カテゴリ: {product_input.category.value}\n"
            f"採寸:\n{_measurement_summary_text(measurements)}\n"
            f"状態:\n{_condition_summary_text(condition_issues)}\n"
            f"補足メモ: {product_input.notes or 'なし'}\n\n"
            "以下のJSON形式のみで出力してください。\n"
            '{"instagram_caption": "...", "ec_description": "...", "hashtags": ["#...", "#..."]}'
        )
        data = claude_client.generate_json(_SYSTEM_PROMPT, user_prompt)
        if data is None:
            return None
        try:
            return GeneratedCopy(
                instagram_caption=data["instagram_caption"],
                ec_description=data["ec_description"],
                hashtags=data.get("hashtags", []),
                generated_by="llm",
            )
        except KeyError:
            return None

    def _generate_fallback(
        self,
        product_input: ProductInput,
        measurements: MeasurementSet | None,
        condition_issues: list[ConditionIssue],
    ) -> GeneratedCopy:
        brand_part = f"{product_input.brand} " if product_input.brand else ""
        caption = (
            f"【入荷】{brand_part}{product_input.product_name}\n"
            f"{_condition_summary_text(condition_issues) if condition_issues else '状態良好な1着です。'}\n"
            "詳細な採寸はプロフィール欄のECサイトをご覧ください。"
        )
        ec_description = (
            f"■商品名\n{brand_part}{product_input.product_name}\n\n"
            f"■採寸\n{_measurement_summary_text(measurements)}\n\n"
            f"■状態\n{_condition_summary_text(condition_issues)}\n\n"
            f"■補足\n{product_input.notes or '特になし'}\n\n"
            "※AIによる自動採寸・自動検品を含みます。実寸には多少の誤差が生じる場合があります。"
        )
        hashtags = ["#古着", "#ヴィンテージ", "#古着女子", "#古着男子"]
        if product_input.brand:
            hashtags.append(f"#{product_input.brand.replace(' ', '')}")
        if product_input.category != GarmentCategory.OTHER:
            hashtags.append(f"#{product_input.category.value}")
        return GeneratedCopy(
            instagram_caption=caption,
            ec_description=ec_description,
            hashtags=hashtags,
            generated_by="template_fallback",
        )
