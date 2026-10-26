"""
models.py
=========
第二部分（对应课程 "Training and validating the model" → "Repeated evaluation
and model stability"）：模型训练、调参、评估。

复现范围
--------
1. stratified k-fold 设置与 random_state 的作用
2. "懒惰模型"（多数类基线）
3. 决策树 + 深度扫描 + 三棵树的树结构可视化
4. 随机森林 + 三参数网格搜索
5. 混淆矩阵与分类型召回率
6. 重复 10 次的交叉验证（稳定性）

所有函数都返回可直接检查的对象（DataFrame / dict / Figure），并同时落盘，
因此在 notebook 里可以边看边讲，在报告里可以直接引用。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

from config import (
    COLOR_FULL,
    COLOR_HALF,
    FEATURES,
    LABELS,
    LABEL_NAMES,
    N_REPEATS,
    N_SPLITS,
    OUT_DIR,
    RANDOM_STATE,
    savefig,
)


# ---------------------------------------------------------------------------
# 交叉验证设置
# ---------------------------------------------------------------------------
def make_skf(random_state: int = RANDOM_STATE, n_splits: int = N_SPLITS) -> StratifiedKFold:
    """分层 k 折：每折内全/半眨眼比例与整体一致。"""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].copy()
    y = df["Blink_type"].copy()
    return X, y


def lazy_baseline(X: pd.DataFrame, y: pd.Series, skf) -> float:
    """多数类基线：一个"什么都不学"的模型能拿到多少准确率。"""
    model = DummyClassifier(strategy="most_frequent")
    scores = cross_val_score(model, X, y, cv=skf)
    return float(scores.mean())


# ---------------------------------------------------------------------------
# 决策树
# ---------------------------------------------------------------------------
def depth_sweep(X: pd.DataFrame, y: pd.Series, skf, max_depth: int = 10) -> pd.DataFrame:
    """扫描 max_depth = 1..max_depth，记录交叉验证准确率。"""
    rows = []
    for depth in range(1, max_depth + 1):
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=depth)
        pred = cross_val_predict(model, X, y, cv=skf)
        rows.append({"max_depth": depth, "accuracy": accuracy_score(y, pred)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "t3_depth_sweep.csv", index=False)
    return table


def plot_depth_sweep(table: pd.DataFrame, lazy_accuracy: float):
    best = table.loc[table["accuracy"].idxmax()]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(table["max_depth"], table["accuracy"], marker="o", color=COLOR_FULL, lw=1.8)
    ax.axhline(
        lazy_accuracy,
        color=COLOR_HALF,
        linestyle="--",
        lw=1.4,
        label=f"Lazy baseline ({lazy_accuracy:.2f})",
    )
    ax.scatter([best["max_depth"]], [best["accuracy"]], s=90, zorder=5,
               facecolor="none", edgecolor=COLOR_FULL, lw=2)
    ax.annotate(
        f"best: depth {int(best['max_depth'])}\nacc {best['accuracy']:.2f}",
        xy=(best["max_depth"], best["accuracy"]),
        xytext=(6, -26),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_title("More complexity does not buy more accuracy")
    ax.set_ylim(0.45, 0.92)
    ax.legend(loc="lower right")
    fig.tight_layout()
    savefig(fig, "fig04_depth_sweep")
    return fig


def tree_complexity_table(X, y, depths=(1, 4, 10)) -> pd.DataFrame:
    """深度 → 叶节点数，用来说明"复杂度"到底涨在哪里。"""
    rows = []
    models = {}
    for depth in depths:
        model = DecisionTreeClassifier(random_state=0, max_depth=depth).fit(X, y)
        models[depth] = model
        rows.append(
            {
                "max_depth": depth,
                "actual_depth": model.get_depth(),
                "n_leaves": model.get_n_leaves(),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "t4_tree_complexity.csv", index=False)
    return table, models


def plot_tree_structure(model, X, name: str, figsize=(14, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(
        model,
        feature_names=list(X.columns),
        class_names=[LABEL_NAMES[c] for c in model.classes_],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=False,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Decision tree, max_depth={model.get_depth()} "
                 f"({model.get_n_leaves()} leaves)")
    fig.tight_layout()
    savefig(fig, name)
    return fig


def narrowest_split(model) -> Dict[str, float]:
    """
    找出树中最窄的阈值区间 —— 用来量化"过拟合"。

    课程给出的例子：0.01667223 < paused_duration <= 0.01668474，区间宽度
    1e-5 s，而 60 fps 下一帧 = 0.0167 s，因此这个切分不可能对应真实的
    行为差异，只是把这一个样本单独切了出去。
    """
    tree = model.tree_
    best = {"width": np.inf}
    feat_names = None  # 由调用方补充；这里只统计阈值宽度
    for node in range(tree.node_count):
        left = tree.children_left[node]
        right = tree.children_right[node]
        if left == -1 or right == -1:
            continue
        thr = tree.threshold[node]
        # 与相邻节点的阈值比较，找出包住本节点的最窄区间
        parent = None
        for cand in range(tree.node_count):
            if tree.children_left[cand] == node:
                parent = tree.threshold[cand]
        if parent is None or not np.isfinite(parent):
            continue
        width = abs(thr - parent)
        if width < best["width"]:
            best = {"width": float(width), "node": node,
                    "feature_index": int(tree.feature[node]),
                    "lower": float(min(thr, parent)),
                    "upper": float(max(thr, parent))}
    best["frame_seconds"] = 1.0 / 59.93
    return best


# ---------------------------------------------------------------------------
# 随机森林
# ---------------------------------------------------------------------------
def default_forest() -> RandomForestClassifier:
    """课程第二段模型：100 棵树、深度 10。"""
    return RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=0, min_samples_leaf=1
    )


def tuned_forest() -> RandomForestClassifier:
    """课程调参后选用的组合：75 棵树、深度 6。"""
    return RandomForestClassifier(
        n_estimators=75, max_depth=6, random_state=0, min_samples_leaf=1
    )


def forest_grid(
    X: pd.DataFrame,
    y: pd.Series,
    skf,
    n_estimators_values=(50, 75, 150),
    max_depth_values=(2, 4, 6, 8, 10),
    min_samples_leaf_values=(1, 3, 5, 7),
) -> pd.DataFrame:
    """穷举 3 × 5 × 4 = 60 种组合（课程里的手工调参）。"""
    rows = []
    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            for min_samples_leaf in min_samples_leaf_values:
                model = RandomForestClassifier(
                    random_state=1,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                )
                pred = cross_val_predict(model, X, y, cv=skf)
                rows.append(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                        "accuracy": accuracy_score(y, pred),
                    }
                )
    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    table.to_csv(OUT_DIR / "t5_forest_grid.csv", index=False)
    return table


def plot_forest_grid(table: pd.DataFrame):
    """网格结果热图：看清"哪一片区域比较稳"。"""
    pivot = (
        table.groupby(["max_depth", "min_samples_leaf"])["accuracy"]
        .max()
        .unstack("min_samples_leaf")
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=0.72, vmax=0.82)
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    ax.set_xlabel("min_samples_leaf")
    ax.set_ylabel("max_depth")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Best CV accuracy per (max_depth, min_samples_leaf)\nover n_estimators")
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    savefig(fig, "fig05_forest_grid_heatmap")
    return fig


# ---------------------------------------------------------------------------
# 混淆矩阵 / 召回率
# ---------------------------------------------------------------------------
def plot_confusion(y, predictions, title: str, name: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        labels=LABELS,
        display_labels=[LABEL_NAMES[c] for c in LABELS],
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    for row in range(2):
        for col in range(2):
            ax.text(col, row + 0.22, "correct" if row == col else "wrong",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if row == col else "crimson")
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    savefig(fig, name)
    return fig


def recall_table(y, preds: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, pred in preds.items():
        rows.append(
            {
                "Model": name,
                "Full blink recall": recall_score(y, pred, pos_label="F"),
                "Half blink recall": recall_score(y, pred, pos_label="H"),
                "Accuracy": accuracy_score(y, pred),
            }
        )
    table = pd.DataFrame(rows)
    display = table.copy()
    for col in ["Full blink recall", "Half blink recall", "Accuracy"]:
        display[col] = (display[col] * 100).round(0).astype(int).astype(str) + "%"
    table.to_csv(OUT_DIR / "t6_recall_table.csv", index=False)
    return display


# ---------------------------------------------------------------------------
# 重复交叉验证 + 稳定性
# ---------------------------------------------------------------------------
def repeated_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], object],
    df: pd.DataFrame,
    n_repeats: int = N_REPEATS,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    用不同的随机划分重复评估同一模型，返回：
      accuracy_scores : 每次重复的准确率
      predictions     : 行级预测表（blink_id / iteration / true / pred / correct）
    """
    scores: List[float] = []
    frames: List[pd.DataFrame] = []
    for i in range(n_repeats):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=i)
        pred = cross_val_predict(model_factory(), X, y, cv=skf)
        scores.append(accuracy_score(y, pred))
        frame = pd.DataFrame(
            {
                "blink_id": df["blink_id"].values,
                "iteration": i,
                "true_label": np.array(y),
                "predicted_label": pred,
            }
        )
        frame["correct_prediction"] = frame["true_label"] == frame["predicted_label"]
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(OUT_DIR / "t7_repeated_predictions.csv", index=False)
    return np.array(scores), predictions


def plot_repeated_scores(scores: np.ndarray, lazy_accuracy: float):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(range(1, len(scores) + 1), scores, color=COLOR_FULL, alpha=0.85)
    ax.axhline(scores.mean(), color="black", lw=1.4,
               label=f"mean = {scores.mean():.3f}")
    ax.fill_between([0.4, len(scores) + 0.6],
                    scores.mean() - scores.std(), scores.mean() + scores.std(),
                    color="grey", alpha=0.2, label=f"±1 sd = {scores.std():.3f}")
    ax.axhline(lazy_accuracy, color=COLOR_HALF, linestyle="--", lw=1.4,
               label=f"lazy baseline = {lazy_accuracy:.2f}")
    ax.set_xlim(0.4, len(scores) + 0.6)
    ax.set_ylim(0.4, 0.95)
    ax.set_xlabel("Repeat (different random 5-fold split)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Repeated 5-fold CV: the score is stable across splits")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig06_repeated_cv")
    return fig
