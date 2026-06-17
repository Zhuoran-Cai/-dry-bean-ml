"""数据预处理展示页。"""

import ui.bootstrap  # noqa: F401

import streamlit as st

from ui.components import feature_tags, info_box, page_header, styled_dataframe
from ui.data import load_class_distribution, load_preprocess_summary
from ui.styles import setup_page

setup_page("数据预处理", "📋")

page_header(
    "PREPROCESSING",
    "数据预处理",
    "对原始干豆数据集进行标签修正、缺失值处理、特征工程与标准化，并划分训练 / 验证 / 测试集。",
)

summary = load_preprocess_summary()
class_dist = load_class_distribution()

st.markdown('<div class="section-label">SUMMARY</div>', unsafe_allow_html=True)
st.markdown("### 预处理摘要")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("训练集", f"{summary['train_samples']:,}")
with c2:
    st.metric("验证集", f"{summary['val_samples']:,}")
with c3:
    st.metric("测试集", f"{summary['test_samples']:,}")
with c4:
    st.metric("去除重复", summary["duplicates_removed"])
with c5:
    st.metric("保留特征", len(summary["selected_features"]))

st.markdown("---")
left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="section-label">PIPELINE</div>', unsafe_allow_html=True)
    st.markdown("### 处理流程")

    steps = [
        ("标签修正", "OCR 误识别字符替换（0→O, 3→E），过滤非法类别"),
        ("数据清洗", "去除缺失值、非法字符，剥离数值单位后缀"),
        (
            "特征工程",
            "构造 5 个比值特征：<br>"
            "① 轴比 AxisRatio<br>"
            "② 面积周长比 AreaPerimeterRatio<br>"
            "③ 凸面积比 ConvexAreaRatio<br>"
            "④ 等效直径比 EquivDiameterRatio<br>"
            "⑤ 圆度紧密度 RoundnessCompactness",
        ),
        ("异常值检测", "IQR 方法检测异常值（仅标记，不删除）"),
        (
            "特征筛选",
            f"依据特征与类别的 Mutual Information、特征间 Pearson 多重共线性过滤，"
            f"最终保留 {len(summary['selected_features'])} 个特征用于建模",
        ),
        ("标准化", "StandardScaler 零均值单位方差缩放"),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(
            f"""
            <div class="panel-card" style="padding:1rem 1.2rem">
                <div style="display:flex;align-items:flex-start;gap:0.75rem">
                    <span style="background:#2E7D6F;color:white;border-radius:50%;
                          width:24px;height:24px;display:inline-flex;align-items:center;
                          justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0">{i}</span>
                    <div>
                        <div style="font-weight:600;color:#1A2332;margin-bottom:0.2rem">{title}</div>
                        <div style="font-size:0.85rem;color:#5C6B7A">{desc}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="section-label">DISTRIBUTION</div>', unsafe_allow_html=True)
    st.markdown("### 类别分布（训练集）")

    chart_df = class_dist.reset_index()
    chart_df.columns = ["类别", "样本数"]
    st.bar_chart(chart_df.set_index("类别"), color="#2E7D6F", height=320)
    styled_dataframe(chart_df, height=220)

st.markdown("---")
st.markdown('<div class="section-label">FEATURES</div>', unsafe_allow_html=True)
st.markdown("### 特征筛选结果")

info_box(
    f"从 <b>{summary['total_features_before_selection']}</b> 个原始/工程特征中，"
    f"依据特征与类别的 Mutual Information、特征间 Pearson 多重共线性过滤，"
    f"最终保留 <b>{len(summary['selected_features'])}</b> 个特征用于建模。"
)

feature_tags(summary["selected_features"])

st.markdown("#### 全部有效类别")
feature_tags(summary["class_names"])
