"""可复用 UI 组件。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def page_header(label: str, title: str, description: str) -> None:
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    st.markdown(f'<p class="section-desc">{description}</p>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, subtitle: str = "", accent: str = "blue") -> None:
    st.markdown(
        f"""
        <div class="kpi-card kpi-accent-{accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_flow() -> None:
    steps = [
        ("原始数据", "7 类干豆形态特征"),
        ("数据预处理", "清洗 · 特征工程 · 标准化"),
        ("特征分析", "IQR · Pearson 热力图 · MI · PCA"),
        ("模型训练", "4 种分类算法对比"),
        ("实验评估", "鲁棒性 · 过拟合 · 混淆矩阵"),
    ]
    parts = ['<div class="pipeline">']
    for i, (title, desc) in enumerate(steps, 1):
        if i > 1:
            parts.append('<div class="pipeline-arrow">→</div>')
        parts.append(
            f"""
            <div class="pipeline-step">
                <div class="step-num">{i}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def show_figure(path: Path | None, caption: str = "", width: int | str = "stretch") -> None:
    if path and path.exists():
        st.image(str(path), width=width)
        if caption:
            st.markdown(f'<p class="figure-caption">{caption}</p>', unsafe_allow_html=True)
    else:
        st.warning(f"图表未找到：{path}")


def styled_dataframe(df: pd.DataFrame, height: int | None = None) -> None:
    kwargs = {"width": "stretch", "hide_index": True}
    if height:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.metric(label, value)


def feature_tags(features: list[str]) -> None:
    tags = "".join(f'<span class="tag">{f}</span>' for f in features)
    st.markdown(f'<div style="line-height:2.2">{tags}</div>', unsafe_allow_html=True)


def info_box(text: str) -> None:
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)
