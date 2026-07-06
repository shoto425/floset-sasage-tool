"""Claude (Anthropic API) の薄いラッパー。

API キー未設定・ネットワーク不可の環境でもパイプライン全体が止まらないよう、
呼び出し失敗時は例外を送出せず `None` を返し、呼び出し元でテンプレート
フォールバックに切り替えられるようにする。
"""
from __future__ import annotations

import base64
import json
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class ClaudeClient:
    def __init__(self) -> None:
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            if not settings.anthropic_api_key:
                return None
            from anthropic import Anthropic

            self._client = Anthropic(api_key=settings.anthropic_api_key)
        return self._client

    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str | None:
        client = self._ensure_client()
        if client is None:
            return None
        try:
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except Exception:
            logger.exception("Claude API呼び出しに失敗しました")
            return None

    def generate_json(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 1024
    ) -> dict | None:
        text = self.generate_text(system_prompt, user_prompt, max_tokens=max_tokens)
        if text is None:
            return None
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Claudeの応答をJSONとして解釈できませんでした: %s", text[:200])
            return None

    def describe_image(
        self, system_prompt: str, user_prompt: str, image_path: str, max_tokens: int = 1024
    ) -> str | None:
        client = self._ensure_client()
        if client is None:
            return None
        try:
            with open(image_path, "rb") as f:
                image_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    }
                ],
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except Exception:
            logger.exception("Claude Vision API呼び出しに失敗しました")
            return None


claude_client = ClaudeClient()
