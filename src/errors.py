"""
errors.py
=========
第三部分（对应课程 "Error analysis: which examples go wrong?" 一节）。

核心问题从"模型多准"转向"模型错在哪儿"：
  1. 哪些眨眼事件被反复判错？（10 次重复 × 5 折 = 每个事件有 10 个预测）
  2. 错误是否集中在某几段视频 / 某几匹马身上？（过度代表检验）
  3. 判错的事件在特征空间里是否与判对的事件分开？（3D 散点）
"""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (注册 3d 投影)

from config import COLOR_ACCENT, COLOR_HALF, FEATURES, OUT_DIR, savefig


def event_error_summary(
    repeated_predictions: pd.DataFrame, threshold: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    统计每个事件被判错的次数。

    返回
    ----
    summary         : 所有被判错过的 blink_id 及其错误次数（降序）
    consistently    : 错误次数 > threshold（"反复判错"的事件）
    """
    preds = repeated_predictions.copy()
    preds["incorrect_prediction"] = ~preds["correct_prediction"]

    times_incorrect = preds.groupby("blink_id")["incorrect_prediction"].sum()
    summary = pd.DataFrame(
        {"blink_id": times_incorrect.index, "times_incorrect": times_incorrect.values}
    )
    summary = summary[summary["times_incorrect"] > 0]
    summary = summary.sort_values("times_incorrect", ascending=False)

    consistently = summary[summary["times_incorrect"] > threshold]
    summary.to_csv(OUT_DIR / "t8_event_error_summary.csv", index=False)
    consistently.to_csv(OUT_DIR / "t9_consistently_misclassified.csv", index=False)
    return summary, consistently


def attach_errors(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """把错误次数合并回事件表，方便后续按视频/马匹/特征空间切片。"""
    out = df.merge(summary[["blink_id", "times_incorrect"]], on="blink_id", how="left")
    out["times_incorrect"] = out["times_incorrect"].fillna(0).astype(int)
    out["status"] = np.where(
        out["times_incorrect"] > 5, "Consistently misclassified", "Other"
    )
    return out


def threshold_sensitivity(summary: pd.DataFrame, n_total: int) -> pd.DataFrame:
    """
    扩展分析：错误阈值如何改变"需要人工复核"的事件数量与构成。

    课程只讨论了 > 5 与 == 10 两种选择；这里把整条曲线画出来，让"选阈值"
    的代价显式化 —— 复核成本 vs 覆盖率。
    """
    rows = []
    for threshold in range(0, 11):
        selected = summary[summary["times_incorrect"] > threshold]
        rows.append(
            {
                "threshold": threshold,
                "n_events_selected": len(selected),
                "pct_of_all_events": round(100 * len(selected) / n_total, 1),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "t10_threshold_sensitivity.csv", index=False)
    return table


def check_overrepresentation(
    all_blinks: pd.DataFrame, problem_blinks: pd.DataFrame, group_column: str
) -> pd.DataFrame:
    """
    检验某段视频 / 某匹马是否"过度出现"在问题事件里。

    关键点：不能只看问题事件的绝对数量，因为事件多的视频本来就会有更多错误。
    因此同时给出：
      expected_problem_blinks = 该组事件数 × 全体问题事件占比
      observed_minus_expected = 观测值 − 期望值（正值 = 高于期望）
      problem_rate            = 组内问题事件比例
    """
    total_counts = all_blinks[group_column].value_counts()
    problem_counts = problem_blinks[group_column].value_counts()

    table = pd.DataFrame(
        {"total_blinks": total_counts, "problem_blinks": problem_counts}
    ).fillna(0)
    table["problem_blinks"] = table["problem_blinks"].astype(int)

    overall_rate = len(problem_blinks) / len(all_blinks)
    table["expected_problem_blinks"] = table["total_blinks"] * overall_rate
    table["observed_minus_expected"] = (
        table["problem_blinks"] - table["expected_problem_blinks"]
    )
    table["problem_rate"] = table["problem_blinks"] / table["total_blinks"]
    table = table.sort_values("observed_minus_expected", ascending=False)
    table.round(2).to_csv(OUT_DIR / f"t11_overrepresentation_{group_column}.csv")
    return table.round(2)


def plot_overrepresentation(table: pd.DataFrame, group_name: str, name: str):
    """观测 − 期望 的偏差图（正=问题偏多，负=问题偏少）。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    values = table["observed_minus_expected"]
    colours = [COLOR_HALF if v > 0 else COLOR_ACCENT for v in values]
    ax.barh(range(len(table)), values, color=colours, alpha=0.85)
    ax.set_yticks(range(len(table)), table.index)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=1)
    for i, (v, rate, n) in enumerate(
        zip(values, table["problem_rate"], table["total_blinks"])
    ):
        ax.text(v + (0.06 if v >= 0 else -0.06), i, f"{rate:.0%} of {int(n)}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("Observed − expected number of problem blinks")
    ax.set_ylabel(group_name)
    ax.set_title(f"Which {group_name.lower()} carry more errors than expected?")
    fig.tight_layout()
    savefig(fig, name)
    return fig


def plot_feature_space_3d(df_with_errors: pd.DataFrame):
    """
    特征空间中的错误分布（课程用 plotly 交互图；这里用 matplotlib 静态图，
    保证在无交互环境/报告中同样可读）。
    """
    fig = plt.figure(figsize=(8.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    ok = df_with_errors[df_with_errors["status"] == "Other"]
    bad = df_with_errors[df_with_errors["status"] == "Consistently misclassified"]

    for label, colour, marker in [("F", "#4c78a8", "o"), ("H", "#f58518", "^")]:
        sub = ok[ok["Blink_type"] == label]
        ax.scatter(
            sub["closing_duration"], sub["paused_duration"], sub["opening_duration"],
            s=22, color=colour, marker=marker, alpha=0.45,
            label=f"{'Full' if label=='F' else 'Half'} (mostly right)",
        )
    ax.scatter(
        bad["closing_duration"], bad["paused_duration"], bad["opening_duration"],
        s=95, facecolor="none", edgecolor="crimson", lw=1.8,
        label="Consistently misclassified (>5/10)",
    )
    ax.set_xlabel("closing_duration")
    ax.set_ylabel("paused_duration")
    ax.set_zlabel("opening_duration")
    ax.set_title("Misclassified blinks sit inside the cloud, not apart from it")
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig11_feature_space_3d")
    return fig


def plot_feature_space_pairgrid(df_with_errors: pd.DataFrame):
    """
    3D 图在静态媒介里很难读，这里补一张两两投影的小多图：
    每一格都是"两个特征"的平面，红圈仍是反复判错的事件。
    判错点若落在两类点的交界处，说明是**特征本身重叠**，换模型也难救。
    """
    pairs = [
        ("closing_duration", "paused_duration"),
        ("closing_duration", "opening_duration"),
        ("paused_duration", "opening_duration"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (fx, fy) in zip(axes, pairs):
        for label, colour, marker in [("F", "#4c78a8", "o"), ("H", "#f58518", "^")]:
            sub = df_with_errors[df_with_errors["Blink_type"] == label]
            ax.scatter(sub[fx], sub[fy], s=24, color=colour, marker=marker,
                       alpha=0.5, label=f"{'Full' if label == 'F' else 'Half'} blink")
        bad = df_with_errors[df_with_errors["status"] == "Consistently misclassified"]
        ax.scatter(bad[fx], bad[fy], s=90, facecolor="none", edgecolor="crimson",
                   lw=1.6, label="Consistently wrong")
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Pairwise projections: the hard events live where the two classes overlap",
                 y=1.02)
    fig.tight_layout()
    savefig(fig, "fig12b_feature_space_pairgrid")
    return fig


def plot_error_composition(df_with_errors: pd.DataFrame):
    """反复判错的事件在两类之间的构成 —— 是否只错半眨眼？"""
    counts = (
        df_with_errors.assign(is_bad=df_with_errors["times_incorrect"] > 5)
        .groupby(["Blink_type", "is_bad"])
        .size()
        .unstack(fill_value=0)
        .reindex(["F", "H"])
    )
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    bottom = np.zeros(2)
    for col, colour, label in [
        (False, "#bcd2e8", "Not consistently wrong"),
        (True, "crimson", "Consistently wrong (>5/10)"),
    ]:
        vals = counts[col].values if col in counts else np.zeros(2)
        ax.bar(["Full blink", "Half blink"], vals, bottom=bottom, color=colour, label=label)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(i, b + v / 2, str(int(v)), ha="center", va="center", fontsize=10)
        bottom += vals
    ax.set_ylabel("Number of events")
    ax.set_title("Consistent errors are a small, mostly half-blink group")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    savefig(fig, "fig12_error_composition")
    return fig
