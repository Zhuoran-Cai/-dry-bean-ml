"""特征分析展示页。"""

import ui.bootstrap  # noqa: F401

import streamlit as st

from ui.components import info_box, page_header, show_figure, styled_dataframe
from ui.data import (
    figure_path,
    load_iqr_report,
    load_mi_report,
    load_pca_report,
    load_preprocess_summary,
)
from ui.styles import setup_page

setup_page("特征分析", "🔬")

page_header(
    "FEATURE ANALYSIS",
    "特征分析",
    "通过 IQR 异常值检测、特征间 Pearson 热力图、Mutual Information 排名与 PCA 降维可视化，理解特征分布与类别可分性。",
)

summary = load_preprocess_summary()
iqr_df = load_iqr_report()
mi_df = load_mi_report()
pca_df = load_pca_report()

st.markdown('<div class="section-label">VISUALIZATION</div>', unsafe_allow_html=True)
st.markdown("### 可视化分析")

tab1, tab2, tab3, tab4 = st.tabs(["IQR 箱线图", "Pearson 热力图", "MI 特征排名", "PCA 二维投影"])

with tab1:
    show_figure(
        figure_path("iqr_boxplots.png"),
        "21 个特征在训练集上的箱线图，用于识别离群点分布",
    )
    info_box(
        "采用 <b>IQR × 1.5</b> 规则检测异常值。本项目仅做检测与报告，"
        "不删除异常样本，以保留数据的完整性。"
    )

with tab2:
    show_figure(
        figure_path("pearson_heatmap.png"),
        "21 个特征之间的 Pearson 相关系数热力图，用于多重共线性分析",
    )

with tab3:
    show_figure(
        figure_path("mi_feature_ranking.png"),
        "各特征与类别的 Mutual Information 得分排名，蓝色为最终入选特征",
    )

with tab4:
    _, chart_col, _ = st.columns([1, 2, 1])
    with chart_col:
        show_figure(
            figure_path("pca_2d_visualization.png"),
            f"基于筛选后 {len(summary['selected_features'])} 维特征的 PCA 二维可视化，颜色区分 7 个豆类别",
        )
    c1, c2 = st.columns(2)
    with c1:
        st.metric("PC1 解释方差", f"{pca_df.iloc[0]['explained_variance_ratio']:.2%}")
    with c2:
        st.metric("PC2 解释方差", f"{pca_df.iloc[1]['explained_variance_ratio']:.2%}")
    st.metric("累计解释方差", f"{pca_df.iloc[1]['cumulative_variance_ratio']:.2%}")

st.markdown("---")
st.markdown('<div class="section-label">REPORTS</div>', unsafe_allow_html=True)
st.markdown("### 分析报告")

tab_a, tab_b, tab_c = st.tabs(["IQR 异常值", "MI 特征排名", "PCA 方差"])

with tab_a:
    display_iqr = iqr_df.copy()
    display_iqr["outlier_ratio"] = display_iqr["outlier_ratio"].map(lambda x: f"{x:.2%}")
    for col in ["Q1", "Q3", "IQR", "lower_bound", "upper_bound"]:
        display_iqr[col] = display_iqr[col].map(lambda x: f"{x:.2f}")
    styled_dataframe(display_iqr, height=400)
    top_outlier = iqr_df.loc[iqr_df["outlier_ratio"].idxmax()]
    info_box(
        f"异常值比例最高的特征为 <b>{top_outlier['feature']}</b>，"
        f"占比 <b>{top_outlier['outlier_ratio']:.2%}</b>（{top_outlier['outlier_count']} 个样本）。"
    )

with tab_b:
    display_mi = mi_df.copy()
    display_mi["mutual_information"] = display_mi["mutual_information"].map(lambda x: f"{x:.4f}")
    display_mi = display_mi.rename(columns={
        "feature": "特征",
        "mutual_information": "互信息 (MI)",
        "selected": "已入选",
    })
    styled_dataframe(display_mi, height=450)
    top_feat = mi_df.iloc[0]
    info_box(
        f"与类别互信息最高的特征为 <b>{top_feat['feature']}</b>，"
        f"MI = <b>{top_feat['mutual_information']:.4f}</b>。"
        f"共筛选 <b>{len(summary['selected_features'])}</b> 个特征进入建模阶段。"
    )

with tab_c:
    display_pca = pca_df.copy()
    display_pca["explained_variance_ratio"] = display_pca["explained_variance_ratio"].map(
        lambda x: f"{x:.2%}"
    )
    display_pca["cumulative_variance_ratio"] = display_pca["cumulative_variance_ratio"].map(
        lambda x: f"{x:.2%}"
    )
    display_pca = display_pca.rename(columns={
        "component": "主成分",
        "explained_variance_ratio": "解释方差比",
        "cumulative_variance_ratio": "累计解释方差比",
    })
    styled_dataframe(display_pca)
