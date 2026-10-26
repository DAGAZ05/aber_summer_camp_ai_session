"""
eda.py
======
第一部分（对应课程 00-worked-example.html 的 "What does the data look like?" 与
"Are the features useful?" 两节）：数据加载 + 探索性可视化。

与课程代码的差别：
  - 全部封装成函数，便于在 notebook 里分节调用、也便于脚本化重跑；
  - 每张图同时保存到 figures/，保证报告里引用的图片可复现；
  - 额外输出一张"总时长"分布图，用于**复述**课程中"用总时长分不开两类"的
    关键结论（课程正文给出了结论，但没有给出对应代码）。
"""

from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    BLINK_EVENTS_CSV,
    COLOR_FULL,
    COLOR_HALF,
    FEATURES,
    LABEL_NAMES,
    OUT_DIR,
    savefig,
)


def load_blink_events() -> pd.DataFrame:
    """读取预处理好的眨眼事件表，并补上总时长与行号（blink_id）。"""
    df = pd.read_csv(BLINK_EVENTS_CSV)
    df = df.reset_index(drop=True)
    df["blink_id"] = df.index
    # 课程里被跳过的"制表"步骤：总时长 = 闭合 + 停顿 + 张开
    df["total_duration"] = (
        df["closing_duration"] + df["paused_duration"] + df["opening_duration"]
    )
    return df


def describe_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """返回数据集的基本描述统计（打印 + 存表）。"""
    info = {
        "n_events": len(df),
        "n_videos": df["video_id"].nunique(),
        "n_horses": df["Horse_ID"].nunique(),
        "counts": df["Blink_type"].value_counts().to_dict(),
        "missing_values": int(df.isna().sum().sum()),
        "duration_summary": df[FEATURES + ["total_duration"]]
        .describe()
        .round(4)
        .to_dict(),
    }
    df[FEATURES + ["total_duration"]].describe().round(4).to_csv(
        OUT_DIR / "t1_feature_describe.csv"
    )
    return info


def plot_total_duration(df: pd.DataFrame):
    """复述课程的关键结论：仅用总时长无法分开全眨眼与半眨眼。"""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(df["total_duration"].min(), df["total_duration"].max(), 40)
    for label, colour in [("F", COLOR_FULL), ("H", COLOR_HALF)]:
        subset = df[df["Blink_type"] == label]
        ax.hist(
            subset["total_duration"],
            bins=bins,
            alpha=0.55,
            label=LABEL_NAMES[label],
            color=colour,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xlabel("Total blink duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Total duration alone does not separate the two blink types")
    ax.legend(title="Blink type")
    fig.tight_layout()
    savefig(fig, "fig01_total_duration_overlap")
    return fig


def plot_feature_distributions(df: pd.DataFrame):
    """三个时长特征的分布对比（课程主图之一）。"""
    x_min = df[FEATURES].min().min()
    x_max = df[FEATURES].max().max()
    shared_bins = np.linspace(x_min, x_max, 51)

    fig, axes = plt.subplots(len(FEATURES), 1, figsize=(9, 10), sharex=True)
    for ax, feature in zip(axes, FEATURES):
        for label, colour in [("F", COLOR_FULL), ("H", COLOR_HALF)]:
            subset = df[df["Blink_type"] == label]
            ax.hist(
                subset[feature],
                bins=shared_bins,
                alpha=0.55,
                label=LABEL_NAMES[label],
                color=colour,
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylabel("Count")
        ax.set_title(feature, loc="left", fontsize=10)
        ax.legend(title="Blink type", loc="upper right")
    axes[-1].set_xlabel("Duration (s)")
    fig.suptitle(
        "Phase durations: all three overlap; no single feature separates the classes",
        y=0.995,
    )
    fig.tight_layout()
    savefig(fig, "fig02_feature_distributions")
    return fig


def overlap_coefficient(df: pd.DataFrame, feature: str) -> float:
    """
    计算两类分布的**重叠系数**（overlap coefficient, OVL）。

    OVL = ∫ min(f_F(x), f_H(x)) dx，取值 0~1：
      0 = 两分布完全分离（该特征极具判别力）
      1 = 两分布完全重合（该特征没有判别力）
    数值上用共享分箱上的直方图概率质量近似。
    """
    vals = df[feature]
    bins = np.linspace(vals.min(), vals.max(), 61)
    p_f, _ = np.histogram(df.loc[df["Blink_type"] == "F", feature], bins=bins, density=True)
    p_h, _ = np.histogram(df.loc[df["Blink_type"] == "H", feature], bins=bins, density=True)
    width = bins[1] - bins[0]
    return float(np.minimum(p_f, p_h).sum() * width)


def feature_separability_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    把"目测重叠程度"量化成一张表 —— 这是复现中的第一处**延伸**。

    课程用直方图"看"分离程度，判断是 opening_duration 分离最明显。但视觉上
    的"分离感"容易被两类样本量与分布形状影响。这里补两个与阈值无关的指标：

      univariate_auc : 单特征 ROC-AUC（把特征当分数直接判别两类）
      overlap_coefficient : 两个分布密度的重叠面积

    注意：paused_duration 被 60 fps 的帧量化（1/60 ≈ 0.0167 s），分布呈离散尖峰，
    会让基于密度的 OVL 偏大/偏小，而基于秩的 AUC 不受影响 —— 因此以 AUC 为准。
    """
    from sklearn.metrics import roc_auc_score

    y = (df["Blink_type"] == "F").astype(int)
    rows = []
    for feature in FEATURES + ["total_duration"]:
        rows.append(
            {
                "feature": feature,
                "univariate_auc": round(float(roc_auc_score(y, df[feature])), 3),
                "overlap_coefficient": round(overlap_coefficient(df, feature), 3),
                "median_F": round(df.loc[df.Blink_type == "F", feature].median(), 4),
                "median_H": round(df.loc[df.Blink_type == "H", feature].median(), 4),
            }
        )
    table = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)
    table.to_csv(OUT_DIR / "t2_feature_separability.csv", index=False)
    return table


def plot_class_balance(df: pd.DataFrame) -> Tuple[object, pd.Series]:
    """类别不平衡可视化。"""
    class_counts = df["Blink_type"].value_counts().reindex(["F", "H"])
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    labels = [LABEL_NAMES[k] for k in class_counts.index]
    bars = ax.bar(labels, class_counts.values, color=[COLOR_FULL, COLOR_HALF], width=0.55)
    for bar, value in zip(bars, class_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value} ({value / len(df):.0%})",
            ha="center",
            fontsize=10,
        )
    ax.set_ylim(0, class_counts.max() * 1.18)
    ax.set_ylabel("Count")
    ax.set_title("Class balance: mild imbalance towards full blinks")
    fig.tight_layout()
    savefig(fig, "fig03_class_balance")
    return fig, class_counts
