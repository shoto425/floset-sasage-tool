"""Sasage AI MVP用UI。

動画をアップロードし、バックエンドAPI (`backend/app/main.py`) を呼び出して
結果（平置き画像・採寸・状態検品・原稿）を確認し、ZIPをダウンロードできる。
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("SASAGE_API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Sasage AI", page_icon="🧥", layout="wide")
st.title("🧥 Sasage AI — 動画1本でささげ業務を自動化")

with st.form("upload_form"):
    video_file = st.file_uploader("商品の動画をアップロード", type=["mp4", "mov", "m4v"])
    col1, col2 = st.columns(2)
    with col1:
        product_name = st.text_input("商品名", placeholder="例: 90s チェックネルシャツ")
        brand = st.text_input("ブランド（任意）")
        category = st.selectbox(
            "カテゴリ", ["tops", "outer", "bottoms", "dress", "other"], index=0
        )
    with col2:
        reference_label = st.text_input("実測した採寸項目名", value="着丈")
        reference_cm = st.number_input("その実測値（cm）", min_value=1.0, value=70.0, step=0.5)
        notes = st.text_area("補足メモ（任意）")

    submitted = st.form_submit_button("処理を開始する")

if submitted:
    if video_file is None or not product_name:
        st.error("動画ファイルと商品名は必須です。")
    else:
        with st.spinner("動画を解析中です…（フレーム抽出→背景除去→採寸→検品→原稿生成）"):
            files = {"video": (video_file.name, video_file.getvalue(), video_file.type)}
            data = {
                "product_name": product_name,
                "category": category,
                "brand": brand,
                "reference_measurement_label": reference_label,
                "reference_measurement_cm": reference_cm,
                "notes": notes,
            }
            try:
                resp = requests.post(f"{API_BASE_URL}/products/process", files=files, data=data, timeout=600)
                resp.raise_for_status()
                st.session_state["result"] = resp.json()
            except requests.RequestException as e:
                st.error(f"処理に失敗しました: {e}")

result = st.session_state.get("result")
if result:
    st.success(f"処理完了: product_id = {result['product_id']}")

    if result.get("warnings"):
        for w in result["warnings"]:
            st.warning(w)

    st.subheader("平置き画像")
    cols = st.columns(min(4, len(result["flat_lay_images"])) or 1)
    for i, image_meta in enumerate(result["flat_lay_images"]):
        image_url = f"{API_BASE_URL}/products/{result['product_id']}/images/{image_meta['file_name']}"
        with cols[i % len(cols)]:
            st.image(image_url, caption=image_meta["file_name"], use_container_width=True)

    st.subheader("採寸データ")
    if result.get("measurements"):
        st.table(
            [
                {"項目": p["label"], "値(cm)": p["value_cm"], "信頼度": p["confidence"]}
                for p in result["measurements"]["points"]
            ]
        )
    else:
        st.info("採寸データがありません。")

    st.subheader("状態検品")
    if result["condition_issues"]:
        for issue in result["condition_issues"]:
            st.write(
                f"- **{issue['location_hint']}**: {issue['description']}"
                f"（深刻度: {issue['severity']} / 接写推奨: {'○' if issue['recommend_closeup'] else '×'}）"
            )
    else:
        st.info("目立った状態の問題は検出されませんでした。")

    st.subheader("原稿案")
    if result.get("generated_copy"):
        st.text_area("Instagram原稿", result["generated_copy"]["instagram_caption"], height=150)
        st.text_area("EC原稿", result["generated_copy"]["ec_description"], height=200)
        st.write(" ".join(result["generated_copy"]["hashtags"]))

    zip_url = f"{API_BASE_URL}/products/{result['product_id']}/zip"
    st.link_button("ZIPをダウンロード", zip_url)
