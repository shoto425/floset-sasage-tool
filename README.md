# floset-sasage-tool（Sasage AI）

「古着ささげ効率化AIツール」— **1本の動画を撮影するだけで**、EC/Instagram向けの
平置き画像・採寸データ・状態検品コメント・原稿・納品ZIPまでをワンストップで
生成する古着・ヴィンテージ服向けツールです。

詳細な設計思想・フォルダ構成・処理フローは [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)、
開発の進捗は [`docs/PROGRESS.md`](docs/PROGRESS.md) を参照してください。

## Phase 1 (MVP) の機能

- 動画からの候補フレーム抽出（鮮明度・構図の多様性で選定）
- 背景除去による白抜き平置き画像の複数生成
- 基準実測値1点からの採寸推定（肩幅・身幅・着丈・袖丈など）
- OpenCVヒューリスティックによる状態検品（傷・色あせ候補の検出＋接写推奨箇所の提示）
- Instagram / EC原稿の自動生成（Claude API連携、未設定時はテンプレートへ自動フォールバック）
- 商品ごとのZIP自動生成（JPEG群 + 採寸データ + 状態テキスト + 原稿案）

## セットアップ

```bash
# バックエンド依存関係
pip install -r backend/requirements.txt
# UI依存関係
pip install -r requirements-ui.txt

cp .env.example .env  # 必要に応じて編集（Claude APIキー等）
```

## 起動

```bash
./scripts/run_dev.sh
```

または個別に:

```bash
# API
cd backend && uvicorn app.main:app --reload --port 8000

# UI（別ターミナル）
streamlit run ui/streamlit_app.py
```

UIは `http://localhost:8501`、APIドキュメント（Swagger）は `http://localhost:8000/docs` で確認できます。

## テスト

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

## ディレクトリ構成

```
backend/app/
├── main.py            # FastAPI エントリポイント
├── core/               # 設定・パイプラインオーケストレーター
├── models/schemas.py   # 中核データモデル
├── services/           # フレーム抽出・背景除去・採寸・検品・原稿生成・梱包の各サービス
├── storage/            # ファイルストレージ抽象化
└── api/routes/         # APIエンドポイント
ui/streamlit_app.py     # MVP用UI
docs/                    # 設計書・進捗報告
```

## ロードマップ（Phase 2 以降）

- 複数アングル動画からの高品質合成
- 動きを活かした状態確認（伸縮時のシワ検出等）
- MediaPipe / Segment Anything による採寸・マスク精度の向上
- Next.js製UIへの移行
