"""動画から平置き画像生成に適した候補フレームを抽出するサービス。

方針:
1. 動画から一定間隔でサンプリングして候補フレーム群を得る。
2. 各フレームの鮮明度(Laplacian分散)を計算し、ブレたフレームを除外する。
3. 知覚ハッシュ(pHash)で構図の近いフレーム同士をまとめ、各クラスタから
   最も鮮明な1枚だけを残す（似た構図を量産しないため）。
4. 鮮明度上位 N 枚を最終候補として返す。

Phase 2 では複数アングル動画（前面/背面/タグ接写など）を区別して扱う
`VideoAnalyzer` に置き換えていく想定。ここでのインターフェースは
「動画パス -> ExtractedFrame のリスト」で固定しておく。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import imagehash
import numpy as np
from PIL import Image

from app.core.config import settings


@dataclass
class ExtractedFrame:
    index: int
    image_bgr: np.ndarray
    sharpness: float
    timestamp_sec: float


def _sharpness_score(gray: np.ndarray) -> float:
    """Laplacian分散でフレームの鮮明度（ブレ・ピント）を評価する。値が大きいほど鮮明。"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _phash(image_bgr: np.ndarray) -> imagehash.ImageHash:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


class FrameExtractionService:
    def __init__(
        self,
        max_candidate_frames: int | None = None,
        max_selected_frames: int | None = None,
        dedupe_hash_distance: int | None = None,
    ) -> None:
        self.max_candidate_frames = max_candidate_frames or settings.max_candidate_frames
        self.max_selected_frames = max_selected_frames or settings.max_selected_frames
        self.dedupe_hash_distance = (
            dedupe_hash_distance
            if dedupe_hash_distance is not None
            else settings.frame_dedupe_hash_distance
        )

    def extract(self, video_path: str) -> list[ExtractedFrame]:
        candidates = self._sample_candidates(video_path)
        if not candidates:
            return []
        deduped = self._dedupe_by_phash(candidates)
        deduped.sort(key=lambda f: f.sharpness, reverse=True)
        return deduped[: self.max_selected_frames]

    def _sample_candidates(self, video_path: str) -> list[ExtractedFrame]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けませんでした: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total_frames <= 0:
            total_frames = self.max_candidate_frames

        step = max(1, total_frames // self.max_candidate_frames)

        candidates: list[ExtractedFrame] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                candidates.append(
                    ExtractedFrame(
                        index=frame_idx,
                        image_bgr=frame,
                        sharpness=_sharpness_score(gray),
                        timestamp_sec=frame_idx / fps,
                    )
                )
            frame_idx += 1
        cap.release()
        return candidates

    def _dedupe_by_phash(self, frames: list[ExtractedFrame]) -> list[ExtractedFrame]:
        kept: list[tuple[ExtractedFrame, imagehash.ImageHash]] = []
        for frame in frames:
            h = _phash(frame.image_bgr)
            match_idx = None
            for i, (kept_frame, kept_hash) in enumerate(kept):
                if h - kept_hash < self.dedupe_hash_distance:
                    match_idx = i
                    break
            if match_idx is None:
                kept.append((frame, h))
            elif frame.sharpness > kept[match_idx][0].sharpness:
                kept[match_idx] = (frame, h)
        return [f for f, _ in kept]
