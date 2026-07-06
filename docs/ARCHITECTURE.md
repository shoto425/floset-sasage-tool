# Sasage AI — アーキテクチャ設計書

「1本の動画を撮影するだけで」平置き画像・採寸・状態検品・原稿・ZIP納品物までワンストップで
生成する、古着・ヴィンテージ服向けささげ業務効率化ツール。

## 1. 設計思想

- **モジュール化**: 各処理（フレーム抽出／背景除去／採寸／検品／原稿生成／梱包）は独立した
  サービスクラスとして実装し、`Pipeline` がオーケストレーションする。個々のサービスは
  差し替え可能なインターフェース（ABC）を持ち、ヒューリスティック実装 → ML実装 → LLM実装へ
  段階的に置き換えられる。
- **Phase 1 は静止画中心**: 動画はまず「複数フレームを含む静止画の集合」として扱う。動画
  ならではの処理（複数アングル合成、動き解析）は Phase 2 で `VideoAnalyzer` として追加する。
- **推論バックエンドの抽象化**: 背景除去・検品・原稿生成はどれも「ヒューリスティック
  (OpenCV) 実装」と「LLM/MLモデル実装」を切り替えられるように `core/config.py` の
  フラグで制御する。ネットワークやAPIキーが無い環境でもヒューリスティックのみで動作する。
- **段階的な精度向上**: 採寸は Phase 1 では「基準となる実測値1点＋マスク形状比率」による
  近似値を出す。Phase 2 で MediaPipe / SAM によるキーポイント検出に置き換える。

## 2. フォルダ構造

```
floset-sasage-tool/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI エントリポイント
│   │   ├── core/
│   │   │   ├── config.py          # 設定（Settings）
│   │   │   └── pipeline.py        # Phase1 パイプラインオーケストレーター
│   │   ├── models/
│   │   │   └── schemas.py         # Pydantic データモデル（本ドキュメントの中核）
│   │   ├── services/              # 各処理ステップ（差し替え可能なサービス群）
│   │   │   ├── frame_extraction.py    # 動画→候補フレーム抽出（鮮明度・多様性選定）
│   │   │   ├── background_removal.py  # 背景除去＋白抜き平置き画像生成
│   │   │   ├── image_processing.py    # 余白調整・リサイズ・JPEG書き出し
│   │   │   ├── measurement.py         # 採寸（肩幅・身幅・着丈・袖丈）
│   │   │   ├── condition_inspection.py# 傷・状態検品
│   │   │   ├── copywriting.py         # EC/Instagram原稿生成
│   │   │   ├── llm_client.py          # Claude(Anthropic) API ラッパー
│   │   │   └── packaging.py           # 商品ごとのZIP生成
│   │   ├── storage/
│   │   │   └── file_store.py      # ローカルファイル保存の抽象化（将来S3化を想定）
│   │   └── api/
│   │       └── routes/
│   │           └── products.py    # /products エンドポイント群
│   ├── tests/                     # pytest ユニットテスト
│   └── requirements.txt
├── ui/
│   └── streamlit_app.py           # MVP用UI（動画アップロード→結果確認→ZIPダウンロード）
├── data/
│   ├── uploads/                   # アップロードされた動画（gitignore対象）
│   └── outputs/                   # 生成物（商品ID単位のディレクトリ、gitignore対象）
├── docs/
│   ├── ARCHITECTURE.md            # 本ドキュメント
│   └── PROGRESS.md                # 進捗報告
├── scripts/
│   └── run_dev.sh                 # 開発用起動スクリプト（API + Streamlit）
└── .env.example
```

## 3. 処理フロー（Phase 1: 静止画中心）

```
[動画ファイル]
     │  (1) FrameExtractionService
     ▼
[候補フレーム群] ── 鮮明度(Laplacian分散) + 構図の多様性(pHash距離)でTop-N選定
     │  (2) BackgroundRemovalService
     ▼
[マスク付き平置き画像] ── 背景除去 → 白背景合成
     │  (3) ImageProcessingService
     ▼
[納品用JPEG群] ── 余白調整・トリミング・リサイズ・命名規則統一
     │
     ├─ (4) MeasurementService ──→ [採寸データ]（肩幅/身幅/着丈/袖丈 + 基準値スケール）
     ├─ (5) ConditionInspectionService ──→ [状態検品テキスト]（症状＋推奨接写箇所）
     └─ (6) CopywritingService（LLM） ──→ [Instagram原稿 / EC原稿]
     │
     ▼
  (7) PackagingService
     ▼
[商品ZIP] = JPEG群 + measurements.json + condition.md + copy.md
```

`core/pipeline.py` の `SasagePipeline.process_video()` が (1)〜(7) を一括実行し、
`ProductResult`（`models/schemas.py`）を返す。各ステップは独立してテスト・差し替え可能。

## 4. 推論バックエンドの抽象化方針

| ステップ | Phase 1 実装 | Phase 2 以降の拡張先 |
|---|---|---|
| 背景除去 | `rembg`（U2Net）→ 失敗時 OpenCV GrabCut にフォールバック | Segment Anything (SAM) による高精度マスク |
| 採寸 | マスク形状＋ユーザー入力の基準実測値1点からの比率推定 | MediaPipe / SAM キーポイント検出、複数アングルからの3D近似 |
| 状態検品 | OpenCV による局所エッジ密度・色ムラ検出のヒューリスティック | 学習済み欠陥検出モデル、Claude Vision による自然文説明の高度化 |
| 原稿生成 | Claude API（`llm_client.py`）にプロンプトテンプレートを渡す | 商品カテゴリ別テンプレート、ブランド別トーン学習 |
| 動画処理 | フレーム抽出のみ（静止画として扱う） | 複数アングル動画の合成、伸縮動作からのシワ・状態確認 |

## 5. Phase 2 以降の拡張ポイント（設計時に確保済み）

- `VideoAnalyzer`（未実装・Phase 2）: 複数アングル動画を受け取り、フレーム間対応から
  3D的な形状復元や「動きを活かした状態確認」（伸縮時のシワ検出）を行う。
  `SasagePipeline` に differential な `process_multi_angle_video()` を追加する形で拡張。
- `storage/file_store.py` はローカルパスのみを扱うインターフェースにしてあるため、
  S3/GCS実装に差し替えてもパイプライン側の変更は不要。
- API は同期処理（アップロード即結果）だが、Phase 2 で動画が長時間・高解像度になった場合は
  ジョブキュー（Celery/RQ）を `pipeline.py` の呼び出し口に追加する想定。

## 6. 進捗報告フォーマット

各フェーズ完了時・主要マイルストーン到達時に、以下フォーマットで `docs/PROGRESS.md` を更新し
チャットでも要約を報告する。

```
## YYYY-MM-DD Phase X 進捗報告

### 完了したこと
- ...

### 動作確認状況
- ...（テスト実行結果・手動確認の範囲）

### 既知の制約・精度に関する注意点
- ...（ヒューリスティックゆえの限界を明示）

### 次のフェーズでやること
- ...

### 確認したいこと（意思決定が必要な事項）
- ...
```
