"""干豆多分类项目 — Streamlit 静态展示入口。"""

import sys
from pathlib import Path

# 确保无论从哪个目录启动，都能正确导入 ui 模块
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from ui.components import feature_tags, info_box, kpi_card, pipeline_flow
from ui.data import (
    load_accuracy_comparison,
    load_experiment_summary,
    load_inference_speed,
    load_preprocess_summary,
)
from ui.styles import setup_page

setup_page("项目概览")

st.markdown(
    '<div class="hero-banner">'
    '<div class="hero-title">干豆多分类机器学习系统</div>'
    '<div class="hero-subtitle">'
    "基于形态学特征的 7 类干豆自动识别 · 数据预处理 → 特征分析 → 多算法训练 → 综合实验评估"
    "</div></div>",
    unsafe_allow_html=True,
)

preprocess = load_preprocess_summary()
summary = load_experiment_summary()
accuracy = load_accuracy_comparison()
best_row = accuracy.loc[accuracy["test_acc"].idxmax()]
speed = load_inference_speed()
fastest_row = speed.loc[speed["inference_ms_per_sample"].idxmin()]

pipeline_flow()

st.markdown("---")
st.markdown('<div class="section-label">OVERVIEW</div>', unsafe_allow_html=True)
st.markdown("### 核心结论")

cols = st.columns(4)
cards = [
    ("最高测试精度", summary["best_test_acc"], f"Test Acc = {best_row['test_acc']:.2%}", "coral"),
    ("最快推理速度", summary["fastest"], f"{fastest_row['inference_ms_per_sample']:.3f} ms/样本", "blue"),
    ("最鲁棒算法", summary["most_robust"].replace("_", " ").title(), "噪声扰动下精度下降最小", "purple"),
    ("最少过拟合", summary["least_overfit"], "训练-测试精度差最小", "teal"),
]
for col, (label, value, sub, accent) in zip(cols, cards):
    with col:
        kpi_card(label, value, sub, accent)

st.markdown("---")
st.markdown('<div class="section-label">DATASET</div>', unsafe_allow_html=True)
st.markdown("### 数据集概况")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("训练集", f"{preprocess['train_samples']:,}")
with c2:
    st.metric("验证集", f"{preprocess['val_samples']:,}")
with c3:
    st.metric("测试集", f"{preprocess['test_samples']:,}")
with c4:
    st.metric("去重样本", preprocess["duplicates_removed"])

info_box(
    f"原始特征 <b>{preprocess['total_features_before_selection']}</b> 维，"
    f"经 Mutual Information 筛选与多重共线性过滤后保留 <b>{len(preprocess['selected_features'])}</b> 维特征。"
    f"共 <b>{len(preprocess['class_names'])}</b> 个类别："
    + "、".join(preprocess["class_names"])
    + "。"
)

st.markdown("#### 筛选特征")
feature_tags(preprocess["selected_features"])

st.markdown("---")
st.markdown('<div class="section-label">NAVIGATION</div>', unsafe_allow_html=True)
st.markdown("### 浏览各模块")
st.markdown(
    "请使用左侧导航栏进入各子页面，查看完整的数据预处理、特征分析、模型训练与实验评估结果。"
)

nav_cols = st.columns(4)
pages = [
    ("📋 数据预处理", "样本清洗、特征工程与类别分布"),
    ("🔬 特征分析", "IQR 异常值、Pearson 热力图、MI 排名与 PCA"),
    ("🤖 模型训练", "四种算法准确率与训练曲线"),
    ("📊 实验评估", "鲁棒性、过拟合与混淆矩阵"),
]
for col, (title, desc) in zip(nav_cols, pages):
    with col:
        st.markdown(
            f'<div class="panel-card"><h4>{title}</h4><p style="color:#5C6B7A;font-size:0.88rem;margin:0">{desc}</p></div>',
            unsafe_allow_html=True,
        )
