"""模型训练展示页。"""

import ui.bootstrap  # noqa: F401

import streamlit as st

from ui.components import info_box, page_header, show_figure, styled_dataframe
from ui.data import figure_path, load_accuracy_comparison
from ui.styles import setup_page

setup_page("模型训练", "🤖")

page_header(
    "MODEL TRAINING",
    "模型训练",
    "在相同预处理数据上训练四种经典分类算法，对比训练集、验证集与测试集精度表现。",
)

accuracy = load_accuracy_comparison()
best = accuracy.loc[accuracy["test_acc"].idxmax()]

st.markdown('<div class="section-label">COMPARISON</div>', unsafe_allow_html=True)
st.markdown("### 算法准确率对比")

show_figure(figure_path("accuracy_comparison.png"), "四种算法在 Train / Val / Test 上的准确率对比")

display_acc = accuracy.copy()
for col in ["train_acc", "val_acc", "test_acc"]:
    display_acc[col] = display_acc[col].map(lambda x: f"{x:.2%}")
display_acc = display_acc.rename(columns={
    "algorithm": "算法标识",
    "name": "算法名称",
    "train_acc": "训练集",
    "val_acc": "验证集",
    "test_acc": "测试集",
})
styled_dataframe(display_acc)

info_box(
    f"测试集精度最高的是 <b>{best['name']}</b>，达到 <b>{best['test_acc']:.2%}</b>。"
    f"四种算法测试精度均超过 91%，说明筛选特征具有良好的类别区分能力。"
)

st.markdown("---")
st.markdown('<div class="section-label">TRAINING CURVES</div>', unsafe_allow_html=True)
st.markdown("### 训练过程曲线")

show_figure(
    figure_path("loss_curves.png"),
    "各算法的训练/验证曲线：逻辑回归 Epoch Loss、KNN K 值扫描、SVM C 参数扫描、随机森林 Staged Loss",
)

st.markdown("#### 算法说明")

algo_info = [
    ("Logistic Regression", "线性多分类，L2 正则化，收敛快、可解释性强"),
    ("KNN", "基于距离的非参数方法，无需显式训练，对局部结构敏感"),
    ("SVM", "RBF 核支持向量机，适合非线性边界，测试精度最高"),
    ("Random Forest", "集成 200 棵决策树，抗过拟合能力强，可输出特征重要性"),
]

cols = st.columns(2)
for i, (name, desc) in enumerate(algo_info):
    with cols[i % 2]:
        st.markdown(
            f'<div class="panel-card"><h4>{name}</h4>'
            f'<p style="color:#5C6B7A;font-size:0.88rem;margin:0">{desc}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown('<div class="section-label">RANKING</div>', unsafe_allow_html=True)
st.markdown("### 测试集精度排名")

ranked = accuracy.sort_values("test_acc", ascending=False).reset_index(drop=True)
ranked.index = ranked.index + 1
for col in ["train_acc", "val_acc", "test_acc"]:
    ranked[col] = ranked[col].map(lambda x: f"{x:.2%}")
ranked = ranked[["name", "train_acc", "val_acc", "test_acc"]].rename(columns={
    "name": "算法",
    "train_acc": "训练集",
    "val_acc": "验证集",
    "test_acc": "测试集",
})
styled_dataframe(ranked)
