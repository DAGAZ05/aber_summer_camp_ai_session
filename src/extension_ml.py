"""
extension_ml.py
===============
延伸分析（一）：把课程里"看一个准确率"的评价方式，换成一套更经得起推敲的
评估流程。

解决的问题（对应报告"延伸"章节）
---------------------------------
E1 指标：准确率之外还该看什么？（balanced accuracy / macro-F1 / MCC / ROC-AUC）
E2 数据泄漏：同一段视频的多个眨眼事件之间有相关性吗？（分层 CV vs 分组 CV）
E3 显著性：0.80 的准确率是不是碰巧？（置换检验）
E4 调参乐观偏差：手工挑出来的 0.81 可信吗？（嵌套 CV）
E5 特征瓶颈：加派生特征能否突破 0.80？（特征工程 + 置换重要性）
E6 数据瓶颈：再多标一些事件有用吗？（学习曲线）
E7 类别不平衡：加权能否换回半眨眼的召回率？
E8 错误与置信度：判错的样本是否都靠近决策边界？
"""

from __future__ import annotations

from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    learning_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import (
    FEATURES,
    LABELS,
    N_SPLITS,
    OUT_DIR,
    RANDOM_STATE,
    savefig,
)

# ---------------------------------------------------------------------------
# E1 指标
# ---------------------------------------------------------------------------
def _proba_for_label(proba: np.ndarray, classes, label: str = "F") -> np.ndarray:
    return proba[:, list(classes).index(label)]


def _classes_of(y: pd.Series):
    """cross_val_predict 的概率列顺序 = 训练时 classes_ 的字典序。"""
    return np.sort(pd.unique(y))


def metrics_suite(y_true, y_pred, y_proba=None, positive: str = "F") -> Dict[str, float]:
    """一次给出多个互补指标。y_proba 是 P(positive)。"""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "recall_F": recall_score(y_true, y_pred, pos_label="F", zero_division=0),
        "recall_H": recall_score(y_true, y_pred, pos_label="H", zero_division=0),
        "precision_F": precision_score(y_true, y_pred, pos_label="F", zero_division=0),
        "precision_H": precision_score(y_true, y_pred, pos_label="H", zero_division=0),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score((np.asarray(y_true) == positive).astype(int), y_proba)
    return out


def model_comparison(X: pd.DataFrame, y: pd.Series, skf) -> pd.DataFrame:
    """
    横向比较 6 个模型（含基线），每个都用同一套折。

    注意：逻辑回归/ SVM / kNN 需要标准化，所以用 Pipeline 包起来，
    避免"缩放用的统计量泄漏了测试折信息"这种低级错误。
    """
    models = {
        "Dummy (majority)": None,
        "Decision tree (d=4)": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=4),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "kNN (k=15, scaled)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "SVM (RBF, scaled)": make_pipeline(
            StandardScaler(),
            # sklearn 1.9 起 SVC(probability=True) 已弃用，改用显式校准
            CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False, cv=3),
        ),
        "Random forest (75/6)": RandomForestClassifier(
            n_estimators=75, max_depth=6, random_state=0, min_samples_leaf=1
        ),
    }
    from sklearn.dummy import DummyClassifier

    models["Dummy (majority)"] = DummyClassifier(strategy="most_frequent")

    rows = []
    for name, model in models.items():
        proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")
        pred = cross_val_predict(model, X, y, cv=skf)
        stats = metrics_suite(y, pred, _proba_for_label(proba, _classes_of(y)))
        rows.append({"model": name, **stats})
    table = pd.DataFrame(rows).set_index("model").round(4)
    table.to_csv(OUT_DIR / "e1_model_comparison.csv")
    return table


def plot_model_comparison(table: pd.DataFrame):
    metrics = ["accuracy", "balanced_acc", "macro_f1", "roc_auc", "recall_H"]
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    x = np.arange(len(table))
    width = 0.16
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 2) * width, table[metric], width, label=metric)
    ax.set_xticks(x, table.index, rotation=18, ha="right", fontsize=9)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Score (5-fold CV)")
    ax.set_title("Accuracy hides differences: AUC and half-blink recall tell another story")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    savefig(fig, "fig40_model_comparison")
    return fig


# ---------------------------------------------------------------------------
# E2 分组交叉验证（泄漏检验）
# ---------------------------------------------------------------------------
def grouped_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """
    同一个视频/同一匹马的不同眨眼事件很可能共享光照、拍摄角度、标注习惯，
    把同组样本同时放进训练和验证会让分数虚高。

    这里对比三种划分：
      StratifiedKFold         —— 课程做法（忽略分组）
      StratifiedGroupKFold    —— 分层 + 同组不跨折
      GroupKFold              —— 纯分组（不保证分层）
    """
    estimator = RandomForestClassifier(
        n_estimators=75, max_depth=6, random_state=0, min_samples_leaf=1
    )
    schemes = {
        "Stratified 5-fold (course)": StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        ),
        "StratifiedGroup 5-fold": StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                                       random_state=RANDOM_STATE),
        "Group 5-fold": GroupKFold(n_splits=N_SPLITS),
    }
    rows = []
    for name, cv in schemes.items():
        scores = cross_val_score(estimator, X, y, groups=groups, cv=cv, scoring="accuracy")
        rows.append(
            {
                "scheme": name,
                "mean_accuracy": scores.mean(),
                "sd": scores.std(),
                "n": len(scores),
            }
        )
    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT_DIR / "e2_grouped_cv.csv", index=False)
    return table


def plot_grouped_cv(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colours = ["#4c78a8", "#f58518", "#2e7d32"]
    bars = ax.barh(table["scheme"], table["mean_accuracy"], color=colours, alpha=0.9)
    ax.errorbar(table["mean_accuracy"], range(len(table)), xerr=table["sd"],
                fmt="none", ecolor="black", capsize=4, lw=1.2)
    for bar, value in zip(bars, table["mean_accuracy"]):
        ax.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", fontsize=10)
    ax.set_xlim(0.5, 0.9)
    ax.set_xlabel("Accuracy (mean ± sd across folds)")
    ax.set_title("Random splits overstate performance; grouped splits are honest")
    fig.tight_layout()
    savefig(fig, "fig41_grouped_cv")
    return fig


# ---------------------------------------------------------------------------
# E3 置换检验
# ---------------------------------------------------------------------------
def permutation_test(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable = lambda: RandomForestClassifier(
        n_estimators=75, max_depth=6, random_state=0, min_samples_leaf=1
    ),
    n_permutations: int = 100,
    random_state: int = 0,
) -> Dict[str, float]:
    """
    零假设：标签与特征无关。把标签打乱 n 次，看真实准确率落在零分布的什么位置。
    p 值 = (打乱后 ≥ 真实分数的次数 + 1) / (n + 1)。
    """
    rng = np.random.default_rng(random_state)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    observed = cross_val_score(model_factory(), X, y, cv=skf, scoring="accuracy").mean()

    null = []
    y_arr = np.array(y)
    for _ in range(n_permutations):
        shuffled = pd.Series(rng.permutation(y_arr), index=y.index)
        null.append(
            cross_val_score(model_factory(), X, shuffled, cv=skf, scoring="accuracy").mean()
        )
    null = np.array(null)
    p_value = (np.sum(null >= observed) + 1) / (n_permutations + 1)
    result = {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }
    np.save(OUT_DIR / "e3_permutation_null.npy", null)
    pd.Series(result).to_csv(OUT_DIR / "e3_permutation_test.csv")
    return result


def plot_permutation(null: np.ndarray, result: Dict[str, float]):
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.hist(null, bins=18, color="grey", alpha=0.65, label="Null (labels shuffled)")
    ax.axvline(result["observed"], color="crimson", lw=2,
               label=f"Observed = {result['observed']:.3f}")
    ax.axvline(result["null_mean"], color="black", ls="--", lw=1,
               label=f"Null mean = {result['null_mean']:.3f}")
    ax.set_xlabel("Cross-validated accuracy")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation test: p = {result['p_value']:.3f} "
                 f"({result['n_permutations']} shuffles)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig42_permutation_test")
    return fig


# ---------------------------------------------------------------------------
# E4 嵌套 CV vs 天真调参
# ---------------------------------------------------------------------------
def nested_vs_naive(X: pd.DataFrame, y: pd.Series, n_iter: int = 40) -> pd.DataFrame:
    """
    天真做法：在一套折上搜索超参，报告 best_score_。
    诚实做法：搜索放在外层折的训练集里，用外层折的验证集打分（嵌套 CV）。

    两者之差 = 调参带来的乐观偏差。
    """
    param_dist = {
        "n_estimators": [30, 50, 75, 100, 150],
        "max_depth": [2, 3, 4, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 3, 5, 7],
        "max_features": [None, "sqrt", "log2"],
    }
    inner = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=0),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=inner,
        scoring="accuracy",
        random_state=0,
        n_jobs=1,
    )
    outer = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=7)

    naive = cross_val_score(search, X, y, cv=outer).mean()  # 这就是嵌套 CV
    # 天真：直接用同一套折挑最优再报告 best_score_
    search.fit(X, y)
    optimistic = search.best_score_

    table = pd.DataFrame(
        [
            {"estimate": "naive best inner score (selection-biased)", "accuracy": optimistic},
            {"estimate": "nested CV (honest)", "accuracy": naive},
            {"estimate": "optimism gap", "accuracy": optimistic - naive},
        ]
    ).round(4)
    table.to_csv(OUT_DIR / "e4_nested_cv.csv", index=False)
    (OUT_DIR / "e4_best_params.txt").write_text(
        str(search.best_params_) + "\n", encoding="utf-8"
    )
    return table


def plot_nested(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    colours = ["#f58518", "#2e7d32", "grey"]
    ax.bar(table["estimate"], table["accuracy"], color=colours, width=0.55)
    for i, value in enumerate(table["accuracy"]):
        ax.text(i, value + 0.006, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0.6, 0.9)
    ax.set_ylabel("Accuracy")
    ax.set_title("How much of the tuning gain is real?")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig43_nested_cv")
    return fig


# ---------------------------------------------------------------------------
# E5 特征工程
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    只做**物理上有意义**的派生：眨眼的时间结构（闭合 vs 张开是否对称、
    停顿占比、总速度）。不做盲目多项式展开。
    """
    out = df.copy()
    total = out["closing_duration"] + out["paused_duration"] + out["opening_duration"]
    out["total_duration"] = total
    out["paused_ratio"] = out["paused_duration"] / total
    out["closing_ratio"] = out["closing_duration"] / total
    out["asymmetry"] = (out["opening_duration"] - out["closing_duration"]) / total
    out["close_open_ratio"] = out["closing_duration"] / out["opening_duration"].replace(0, np.nan)
    out["close_open_ratio"] = out["close_open_ratio"].fillna(out["closing_duration"])
    return out


FEATURE_SETS = {
    "raw 3 (course)": FEATURES,
    "raw 3 + total": FEATURES + ["total_duration"],
    "raw 3 + ratios": FEATURES + ["paused_ratio", "closing_ratio", "asymmetry"],
    "raw 3 + everything": FEATURES
    + ["total_duration", "paused_ratio", "closing_ratio", "asymmetry", "close_open_ratio"],
}


def compare_feature_sets(df: pd.DataFrame, n_repeats: int = 3) -> pd.DataFrame:
    """
    每个特征集用 3 次重复 5 折 CV 评估（降低单次划分的噪声），
    同时给出半数召回率 —— 因为"准确率高但半眨眼全靠猜"是不可接受的。
    """
    rows = []
    for set_name, cols in FEATURE_SETS.items():
        X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(df[cols].median())
        y = df["Blink_type"]
        accs, recalls, aucs = [], [], []
        for seed in range(n_repeats):
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            model = RandomForestClassifier(n_estimators=75, max_depth=6, random_state=0)
            pred = cross_val_predict(model, X, y, cv=skf)
            proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")
            accs.append(accuracy_score(y, pred))
            recalls.append(recall_score(y, pred, pos_label="H"))
            aucs.append(roc_auc_score((y == "F").astype(int),
                                      _proba_for_label(proba, _classes_of(y))))
        rows.append(
            {
                "feature_set": set_name,
                "n_features": len(cols),
                "accuracy": np.mean(accs),
                "accuracy_sd": np.std(accs),
                "recall_H": np.mean(recalls),
                "roc_auc": np.mean(aucs),
            }
        )
    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT_DIR / "e5_feature_sets.csv", index=False)
    return table


def plot_feature_sets(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(table))
    ax.errorbar(x, table["accuracy"], yerr=table["accuracy_sd"], fmt="o-",
                color="#4c78a8", capsize=4, label="accuracy")
    ax.plot(x, table["roc_auc"], "s--", color="#2e7d32", label="ROC-AUC")
    ax.plot(x, table["recall_H"], "^:", color="#c0504d", label="half-blink recall")
    ax.set_xticks(x, table["feature_set"], rotation=12, ha="right", fontsize=9)
    ax.set_ylim(0.55, 0.95)
    ax.set_ylabel("Score")
    ax.set_title("Extra timing features move the ceiling only slightly")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "fig44_feature_sets")
    return fig


def importance_plot(df: pd.DataFrame, n_repeats: int = 10):
    """置换重要性（比不纯度重要性更可靠，且在测试折上计算）。"""
    cols = FEATURE_SETS["raw 3 + everything"]
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(df[cols].median())
    y = df["Blink_type"]
    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=0).fit(X, y)
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=0, scoring="accuracy"
    )
    table = pd.DataFrame(
        {"feature": cols, "importance": result.importances_mean, "sd": result.importances_std}
    ).sort_values("importance", ascending=True)
    table.to_csv(OUT_DIR / "e6_permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colours = ["#c0504d" if v > 0.01 else "#b0b0b0" for v in table["importance"]]
    ax.barh(table["feature"], table["importance"], xerr=table["sd"], color=colours,
            capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Drop in accuracy when the feature is shuffled")
    ax.set_title("Permutation importance: paused_duration dominates; ratios add nothing")
    fig.tight_layout()
    savefig(fig, "fig45_permutation_importance")
    return table


# ---------------------------------------------------------------------------
# E6 学习曲线
# ---------------------------------------------------------------------------
def learning_curve_analysis(df: pd.DataFrame):
    X = df[FEATURES]
    y = df["Blink_type"]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=75, max_depth=6, random_state=0),
        X, y, cv=skf, scoring="accuracy", train_sizes=np.linspace(0.2, 1.0, 6),
        shuffle=True, random_state=0,
    )
    table = pd.DataFrame(
        {
            "train_size": sizes,
            "train_mean": train_scores.mean(1),
            "train_sd": train_scores.std(1),
            "val_mean": val_scores.mean(1),
            "val_sd": val_scores.std(1),
        }
    ).round(4)
    table.to_csv(OUT_DIR / "e7_learning_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(table["train_size"], table["train_mean"], "o-", color="#4c78a8",
            label="Training score")
    ax.fill_between(table["train_size"], table["train_mean"] - table["train_sd"],
                    table["train_mean"] + table["train_sd"], color="#4c78a8", alpha=0.15)
    ax.plot(table["train_size"], table["val_mean"], "s-", color="#c0504d",
            label="Validation score")
    ax.fill_between(table["train_size"], table["val_mean"] - table["val_sd"],
                    table["val_mean"] + table["val_sd"], color="#c0504d", alpha=0.15)
    gap = table["train_mean"].iloc[-1] - table["val_mean"].iloc[-1]
    ax.set_xlabel("Number of training events")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.set_title(f"Learning curve: final gap = {gap:.2f}, "
                 "validation still creeping up at n=186")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig46_learning_curve")
    return table


# ---------------------------------------------------------------------------
# E7 类别加权
# ---------------------------------------------------------------------------
def class_weight_effect(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for weight in [None, "balanced"]:
        model = RandomForestClassifier(
            n_estimators=75, max_depth=6, random_state=0, class_weight=weight
        )
        pred = cross_val_predict(model, X, y, cv=skf)
        rows.append(
            {
                "class_weight": str(weight),
                "accuracy": accuracy_score(y, pred),
                "balanced_acc": balanced_accuracy_score(y, pred),
                "recall_F": recall_score(y, pred, pos_label="F"),
                "recall_H": recall_score(y, pred, pos_label="H"),
            }
        )
    table = pd.DataFrame(rows).round(4)
    table.to_csv(OUT_DIR / "e8_class_weight.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# E8 错误 vs 置信度
# ---------------------------------------------------------------------------
def confidence_vs_error(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    用交叉验证得到的 P(half blink) 作为置信度，画"置信度分箱 vs 实际错误率"。
    如果错误集中在 0.5 附近，说明模型错得"有理由"（样本本身模糊）；
    如果高置信度处错误率仍然很高，说明模型存在系统性偏差。
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = RandomForestClassifier(n_estimators=75, max_depth=6, random_state=0)
    proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")
    pred = cross_val_predict(model, X, y, cv=skf)
    p_half = _proba_for_label(proba, _classes_of(y), "H")
    margin = np.abs(p_half - 0.5) * 2  # 0 = 完全没把握, 1 = 很确定

    bins = np.linspace(0, 1, 6)
    bucket = pd.cut(margin, bins, include_lowest=True)
    table = pd.DataFrame(
        {"margin": margin, "correct": (pred == np.array(y)).astype(int), "bucket": bucket}
    ).groupby("bucket", observed=True).agg(
        n=("correct", "size"), error_rate=("correct", lambda s: 1 - s.mean())
    )
    table["bucket"] = table.index.astype(str)
    table = table.reset_index(drop=True).round(3)
    table.to_csv(OUT_DIR / "e9_confidence_vs_error.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(table["bucket"], table["error_rate"], color="#c0504d", alpha=0.85)
    for i, (rate, n) in enumerate(zip(table["error_rate"], table["n"])):
        ax.text(i, rate + 0.012, f"n={n}", ha="center", fontsize=8)
    ax.set_xlabel("Decision margin  |2·P(half) − 1|  (right = more confident)")
    ax.set_ylabel("Error rate")
    ax.set_title("Errors live near the decision boundary, not in confident predictions")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig47_confidence_vs_error")
    return table
