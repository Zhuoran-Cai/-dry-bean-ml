# 🫘 干豆多分类机器学习系统

> 基于形态学特征的 **7 类干豆** 自动识别 · 完整数据预处理 → 特征分析 → 多算法训练 → 综合实验评估

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 目录

- [项目简介](#-项目简介)
- [展示界面](#-展示界面)
- [数据集描述](#-数据集描述)
- [数据处理流程](#-数据处理流程)
- [算法实现](#-算法实现)
- [模型精度对比](#-模型精度对比)
- [实验结论](#-实验结论)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [引用与致谢](#-引用与致谢)

---

## ✨ 项目简介

本项目面向**干豆（Dry Bean）形态学多分类**任务，构建了一套从原始 CSV 数据到可视化展示的完整机器学习流水线。系统对含 OCR 噪声的「脏数据」进行清洗与特征工程，训练并对比 **4 种经典分类算法**，从准确率、推理速度、噪声鲁棒性与过拟合程度等维度进行综合评估，并通过 **Streamlit** 交互式仪表盘呈现全部实验结果。

**核心亮点：**

- 端到端可复现流水线，配置驱动（YAML），一键训练与评估
- 21 维原始/工程特征 → MI 筛选 + 共线性过滤 → **13 维**有效特征
- 四种算法统一接口，自动生成对比图表与 CSV 报告
- 精美 Web 展示界面，适合课程答辩与 GitHub 作品集展示

---

## 🖥️ 展示界面

启动 Streamlit 后，系统提供 **5 个页面**，覆盖项目全貌：

| 页面 | 内容 |
|------|------|
| **项目概览** | 核心 KPI、数据集概况、导航入口 |
| **数据预处理** | 清洗流程、类别分布、特征列表 |
| **特征分析** | IQR 箱线图、Pearson 热力图、MI 排名、PCA 二维可视化 |
| **模型训练** | 四算法准确率对比、Loss / 参数扫描曲线 |
| **实验评估** | 推理速度、噪声鲁棒性、过拟合分析、混淆矩阵、特征重要性 |

```bash
# Windows
run_ui.bat

# 或通用命令
streamlit run app.py --server.port 8505
```

浏览器访问 `http://localhost:8505` 即可查看完整展示界面。

### 界面预览

<p align="center">
  <img src="results/figures/accuracy_comparison.png" alt="准确率对比" width="45%" />
  <img src="results/figures/loss_curves.png" alt="训练曲线" width="45%" />
</p>

<p align="center">
  <img src="results/figures/inference_speed.png" alt="推理速度" width="45%" />
  <img src="results/figures/robustness.png" alt="鲁棒性实验" width="45%" />
</p>

<p align="center">
  <img src="results/figures/confusion_matrices.png" alt="混淆矩阵" width="90%" />
</p>

---

## 📊 数据集描述

本项目使用 **Dry Bean Dataset（Dirty 版本）**——在标准 UCI 干豆数据集基础上引入标签 OCR 误识别、缺失值与非法字符等噪声，更贴近真实工业场景。

| 属性 | 说明 |
|------|------|
| **任务类型** | 7 分类（多分类） |
| **样本总量** | 13,587（训练 9,503 · 验证 1,347 · 测试 2,737） |
| **原始特征** | 16 个形态学特征（面积、周长、长轴、偏心率、圆度、紧密度等） |
| **工程特征** | 5 个比值/组合特征（轴比、凸包比、等效直径比等） |
| **筛选后特征** | **13 维**（Mutual Information + 多重共线性过滤） |
| **类别标签** | 7 类土耳其产干豆品种 |

**7 个类别：**

`BARBUNYA` · `BOMBAY` · `CALI` · `DERMASON` · `HOROZ` · `SEKER` · `SIRA`

**原始特征示例：**

Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1–4

---

## 🔧 数据处理流程

```mermaid
flowchart LR
    A[原始 CSV] --> B[标签修正]
    B --> C[数值清洗]
    C --> D[缺失值填充]
    D --> E[特征工程]
    E --> F[重复样本去除]
    F --> G[MI 特征筛选]
    G --> H[共线性过滤]
    H --> I[StandardScaler 标准化]
    I --> J[训练 / 验证 / 测试集]
```

| 步骤 | 方法 | 说明 |
|------|------|------|
| 标签修正 | OCR 字符替换 | `0→O`、`3→E`，统一大写，过滤非法类别 |
| 数据清洗 | 正则 + 类型转换 | 去除 `"?"`、空字符串，剥离 `" cm"` 等单位后缀 |
| 缺失值处理 | `SimpleImputer` | 以训练集中位数填充，避免数据泄漏 |
| 特征工程 | 比值特征构造 | AxisRatio、AreaPerimeterRatio、ConvexAreaRatio、EquivDiameterRatio、RoundnessCompactness |
| 异常值分析 | IQR（×1.5） | 箱线图可视化；默认不删除，保留样本量 |
| 特征筛选 | Mutual Information | 保留 MI > 0.05 的特征 |
| 共线性过滤 | Pearson 相关系数 | 剔除 \|r\| > 0.95 的冗余特征对 |
| 标准化 | `StandardScaler` | 仅在训练集 fit，同步变换验证/测试集 |
| 降维可视化 | PCA（n=2） | 用于特征分析页面的二维散点图 |

**最终保留的 13 个特征：**

`Area` · `Perimeter` · `MajorAxisLength` · `Extent` · `roundness` · `Compactness` · `ShapeFactor1` · `ShapeFactor2` · `ShapeFactor4` · `AxisRatio` · `ConvexAreaRatio` · `EquivDiameterRatio` · `RoundnessCompactness`

---

## 🤖 算法实现

系统实现了 **4 种** scikit-learn 分类器，统一训练 / 评估 / 持久化接口：

| 算法 | 类型 | 关键超参数 | 训练曲线 |
|------|------|-----------|---------|
| **Logistic Regression** | 线性模型 | C=1.0, solver=lbfgs | SGD 逐步 Log Loss |
| **KNN** | 距离模型 | k=5, weights=distance | k 值扫描（1–31） |
| **SVM** | 核方法 | RBF 核, C=10, γ=scale | C 参数扫描 |
| **Random Forest** | 集成学习 | 200 棵树, 不限深度 | 逐步增加树数量的 Staged Loss |

**扩展实验：**

- **推理速度**：逐样本 `predict` 计时（100 次重复取均值），反映在线推理延迟
- **鲁棒性**：对标准化测试集注入高斯 / 均匀 / 椒盐噪声，评估精度下降
- **过拟合分析**：对比训练集与测试集精度差
- **混淆矩阵 & 特征重要性**：逐类诊断 + 随机森林 SHAP 替代（Gini 重要性）

---

## 📈 模型精度对比

> 以下结果为当前流水线完整运行（`python main.py --task all`）后在**独立测试集**上的指标。

### 综合精度表

| 算法 | 训练精度 | 验证精度 | 测试精度 | 过拟合间隙 ↓ | 推理耗时 (ms/样本) ↓ |
|:----:|:--------:|:--------:|:--------:|:------------:|:-------------------:|
| **SVM** | 93.75% | 92.20% | **92.95%** | 0.80% | 0.388 |
| **KNN** | 91.80% | 91.17% | 92.29% | **−0.49%** | 0.633 |
| **Random Forest** | 100.00% | 92.06% | 92.47% | 7.53% | 0.132 |
| **Logistic Regression** | 92.46% | 91.76% | 91.60% | 0.86% | **0.065** |

> **说明：**
> - KNN 训练精度采用**排除自身近邻**的评估方式，避免 KNN 在训练集上虚高至 100%
> - 过拟合间隙 = 训练精度 − 测试精度；负值表示测试集表现优于训练集（轻微欠拟合或统计波动）
> - 推理耗时为单样本在线预测平均延迟，逻辑回归最快

### 指标解读

| 维度 | 最优算法 | 结论 |
|------|---------|------|
| 测试精度 | **SVM（92.95%）** | RBF 核 SVM 在本数据集上综合分类能力最强 |
| 推理速度 | **Logistic Regression（0.065 ms）** | 线性模型结构简单，适合实时部署 |
| 噪声鲁棒性 | **Random Forest** | 三种噪声扰动下平均精度下降最小 |
| 泛化稳定性 | **KNN（过拟合间隙 −0.49%）** | 训练-测试精度最为接近 |

---

## 🏆 实验结论

1. **SVM** 在测试集上取得最高精度（92.95%），是精度优先场景的首选。
2. **Logistic Regression** 推理速度最快（0.065 ms/样本），精度仍维持在 91.6% 以上，适合对延迟敏感的场景。
3. **Random Forest** 训练精度达 100%，存在一定过拟合（间隙 7.5%），但在噪声环境下表现最稳健。
4. **KNN** 测试精度 92.29%，k=5 + 距离加权配置下泛化良好；单样本推理较慢（需检索近邻）。
5. 高斯/均匀噪声对精度影响温和（≤7%），椒盐噪声破坏性更强，已通过独立较低档位（1%–5%）进行可控评估。

---

## 📁 项目结构

```
ML_Bean/
├── app.py                  # Streamlit 展示入口（项目概览）
├── main.py                 # 命令行统一入口（预处理 / 训练 / 评估）
├── run_ui.bat              # Windows 一键启动展示界面
├── requirements.txt        # Python 依赖
├── config/
│   ├── preprocess.yaml     # 预处理配置
│   └── train.yaml          # 训练与实验配置
├── data/                   # 原始数据集（train / val / test）
├── prep/                   # 数据加载与预处理模块
├── src/                    # 四种算法 + 绑图 + 推理计时
├── train_eval/             # 训练与评估流水线
├── pages/                  # Streamlit 子页面（预处理 / 特征 / 训练 / 评估）
├── ui/                     # 展示界面组件与样式
├── models/trained/         # 训练好的模型（.pkl）
└── results/
    ├── processed/          # 预处理后 CSV
    ├── experiments/        # 精度 / 鲁棒性 / 推理速度 CSV 报告
    └── figures/            # 全部实验图表（PNG）
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/ML_Bean.git
cd ML_Bean
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行完整流水线

```bash
# 全流程：预处理 → 特征分析 → 训练 → 评估
python main.py --task all

# 也可分步执行
python main.py --task preprocess   # 数据预处理
python main.py --task analyze      # IQR / Pearson / MI / PCA
python main.py --task train        # 训练四种算法
python main.py --task evaluate     # 测试集评估与图表生成
```

### 4. 启动展示界面

```bash
streamlit run app.py --server.port 8505
```

---

## 📎 引用与致谢

- 数据集基于 [UCI Dry Bean Dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)（Koklu, M. & Ozkan, I.A., 2020）
- 本项目为「机器学习与项目实践」课程大作业，涵盖数据预处理、特征工程、多算法对比与可视化展示

---

<p align="center">
  <sub>如果这个项目对你有帮助，欢迎 ⭐ Star 支持一下</sub>
</p>
