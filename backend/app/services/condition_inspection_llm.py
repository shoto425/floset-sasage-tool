"""Claude Vision を用いた状態検品（Phase 2 想定・オプトイン実装）。

`SASAGE_CONDITION_BACKEND=llm_vision` かつ `SASAGE_ANTHROPIC_API_KEY` が
設定されている場合のみ有効になる。ヒューリスティック実装と同じ
`ConditionInspector` インターフェースを満たすため、パイプライン側の
コード変更なしに切り替えられる。
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from app.models.schemas import ConditionIssue, ConditionSeverity
from app.services.condition_inspection import ConditionInspector
from app.services.llm_client import claude_client

_SYSTEM_PROMPT = (
    "あなたは古着・ヴィンテージ衣類の検品を行う専門スタッフです。"
    "画像から傷・汚れ・色落ち・ほつれ・生地の薄くなり等の状態を具体的に指摘し、"
    "接写確認を推奨する箇所も併せて示してください。"
)

_USER_PROMPT = (
    "この平置き画像を検品し、検出した問題点を次のJSON配列の形式だけで出力してください。"
    '[{"description": "袖口のほつれ", "severity": "low|medium|high", '
    '"location_hint": "左袖口", "recommend_closeup": true}]\n'
    "問題が無ければ空配列 [] を返してください。JSON以外の文章は出力しないでください。"
)


class LLMConditionInspector(ConditionInspector):
    def inspect(self, image_bgr: np.ndarray, mask: np.ndarray) -> list[ConditionIssue]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "frame.jpg"
            cv2.imwrite(str(tmp_path), image_bgr)
            raw = claude_client.describe_image(_SYSTEM_PROMPT, _USER_PROMPT, str(tmp_path))

        if raw is None:
            return []
        try:
            start, end = raw.index("["), raw.rindex("]") + 1
            items = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return []

        issues: list[ConditionIssue] = []
        for item in items:
            try:
                issues.append(
                    ConditionIssue(
                        description=item["description"],
                        severity=ConditionSeverity(item.get("severity", "low")),
                        location_hint=item.get("location_hint", "不明"),
                        recommend_closeup=item.get("recommend_closeup", True),
                        detected_by="llm_vision",
                    )
                )
            except (KeyError, ValueError):
                continue
        return issues
