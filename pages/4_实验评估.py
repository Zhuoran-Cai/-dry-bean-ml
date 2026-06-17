"""实验评估展示页。"""

import ui.bootstrap  # noqa: F401

import streamlit as st

from ui.components import info_box, kpi_card, page_header, show_figure, styled_dataframe
from ui.data import (
    figure_path,
    load_experiment_summary,
    load_inference_speed,
    load_overfitting,
    load_robustness,
)
from ui.styles import setup_page

setup_page("实验评估", "📊")

page_header(
    "EVALUATION",
    "实验评估",
    "从推理速度、噪声鲁棒性、过拟合程度、混淆矩阵与特征重要性等维度，全面评估模型表现。",
)

summary = load_experiment_summary()
overfit = load_overfitting()
robust = load_robustness()
speed = load_inference_speed()

st.markdown('<div class="section-label">HIGHLIGHTS</div>', unsafe_allow_html=True)
st.markdown("### 实验结论")

cols = st.columns(4)
cards = [
    ("最高测试精度", summary["best_test_acc"], "综合准确率最优", "coral"),
    ("最快推理", summary["fastest"], "单样本推理耗时最短", "blue"),
    ("最鲁棒", summary["most_robust"].replace("_", " ").title(), "噪声环境下精度最稳定", "purple"),
    ("最少过拟合", summary["least_overfit"], "泛化能力最佳", "teal"),
]
for col, (label, value, sub, accent) in zip(cols, cards):
    with col:
        kpi_card(label, value, sub, accent)

st.markdown("---")
st.markdown('<div class="section-label">PERFORMANCE</div>', unsafe_allow_html=True)
st.markdown("### 推理速度与过拟合")

col_l, col_r = st.columns(2)
with col_l:
    show_figure(figure_path("inference_speed.png"), "单样本平均推理耗时（ms）")
    display_speed = speed.copy()
    display_speed["inference_ms_per_sample"] = display_speed["inference_ms_per_sample"].map(
        lambda x: f"{x:.4f}"
    )
    display_speed = display_speed.rename(columns={
        "algorithm": "算法标识",
        "name": "算法名称",
        "inference_ms_per_sample": "推理耗时 (ms)",
    })
    styled_dataframe(display_speed)

with col_r:
    show_figure(figure_path("overfitting.png"), "训练集与测试集精度差（过拟合间隙）")
    display_overfit = overfit.copy()
    for col in ["train_acc", "test_acc", "overfit_gap"]:
        display_overfit[col] = display_overfit[col].map(lambda x: f"{x:.2%}")
    display_overfit = display_overfit.rename(columns={
        "algorithm": "算法标识",
        "name": "算法名称",
        "train_acc": "训练集精度",
        "test_acc": "测试集精度",
        "overfit_gap": "过拟合间隙",
    })
    styled_dataframe(display_overfit)

least_gap = overfit.loc[overfit["overfit_gap"].idxmin()]
info_box(
    f"<b>{least_gap['name']}</b> 的过拟合间隙最小（<b>{least_gap['overfit_gap']:.2%}</b>），"
    f"泛化表现最为稳定。"
)

st.markdown("---")
st.markdown('<div class="section-label">ROBUSTNESS</div>', unsafe_allow_html=True)
st.markdown("### 鲁棒性实验")

show_figure(
    figure_path("robustness.png"),
    "在标准化后的测试集注入高斯 / 均匀 / 椒盐噪声，评估已训练模型的推理鲁棒性",
)

avg_drop = robust.groupby("algorithm")["acc_drop"].mean().sort_values()
st.markdown("#### 各算法平均精度下降")
drop_df = avg_drop.reset_index()
drop_df.columns = ["算法", "平均精度下降"]
drop_df["平均精度下降"] = drop_df["平均精度下降"].map(lambda x: f"{x:.2%}")
styled_dataframe(drop_df)

most_robust = avg_drop.idxmin()
info_box(
    f"在三种噪声、四个强度共 12 组扰动下，<b>{most_robust.replace('_', ' ').title()}</b> "
    f"的平均精度下降最小（<b>{avg_drop[most_robust]:.2%}</b>），鲁棒性最佳。"
)

st.markdown("---")
st.markdown('<div class="section-label">DIAGNOSTICS</div>', unsafe_allow_html=True)
st.markdown("### 诊断分析")

tab1, tab2 = st.tabs(["混淆矩阵", "特征重要性"])

with tab1:
    show_figure(
        figure_path("confusion_matrices.png"),
        "四种算法在测试集上的混淆矩阵，对角线越深表示分类越准确",
    )

with tab2:
    show_figure(
        figure_path("rf_feature_importance.png"),
        "随机森林模型的特征重要性排序，反映各特征对分类决策的贡献",
    )
