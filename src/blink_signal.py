"""
signal.py
=========
第五部分（对应 final_session 与课程 "Comparing the signals" 一节）：把眼区裁片
压成一条亮度信号，与人工标注的真值对照。

这里同时引入"信号好不好"的可计算指标（频帧级 ROC-AUC、阈值扫描、事件匹配），
把课程里"用眼睛看是否对齐"升级为"用数字说对齐了多少"。
"""

from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import COLOR_SIGNAL, COLOR_TRUTH, GROUNDTRUTH_NPY, OUT_DIR, savefig


# ---------------------------------------------------------------------------
# 从眼区裁片视频提取亮度信号（课程 final_session 的第一步）
# ---------------------------------------------------------------------------
def extract_mean_brightness(video_path) -> np.ndarray:
    """
    逐帧读取眼区裁片视频，返回每帧的平均亮度（灰度）。

    这就是课程 final_session 第一段代码的等价实现：一个视频 → 一条一维信号。
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    values = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        values.append(float(gray.mean()))
    cap.release()
    return np.asarray(values)


# ---------------------------------------------------------------------------
# 真值
# ---------------------------------------------------------------------------
def load_groundtruth() -> np.ndarray:
    return np.load(GROUNDTRUTH_NPY).astype(int).ravel()


def groundtruth_intervals(gt: np.ndarray) -> pd.DataFrame:
    """把 0/1 序列转成 [start, end) 区间，方便数事件、算时长。"""
    padded = np.concatenate([[0], gt, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return pd.DataFrame(
        {
            "event": np.arange(1, len(starts) + 1),
            "start": starts,
            "end": ends,
            "length_frames": ends - starts,
        }
    )


def plot_groundtruth(gt: np.ndarray, fps: float):
    events = groundtruth_intervals(gt)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.step(np.arange(len(gt)), gt, where="post", color=COLOR_TRUTH, lw=1.2)
    ax.fill_between(np.arange(len(gt)), 0, gt, step="post", color=COLOR_TRUTH, alpha=0.18)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Blink signal")
    ax.set_yticks([0, 1], ["no blink", "blink"])
    ax.set_ylim(-0.1, 1.15)
    ax.set_title(
        f"Ground-truth annotations: {len(events)} blink events, "
        f"median duration {events.length_frames.median() / fps:.2f} s"
    )
    fig.tight_layout()
    savefig(fig, "fig30_groundtruth")
    return fig, events


# ---------------------------------------------------------------------------
# 去趋势
# ---------------------------------------------------------------------------
def detrend(signal: pd.Series, fps: float, seconds: float) -> Tuple[pd.Series, pd.Series]:
    """返回 (baseline, detrended)。窗口按帧数换算，且强制为奇数以便居中。"""
    window = max(3, int(fps * seconds))
    if window % 2 == 0:
        window += 1
    baseline = signal.rolling(window=window, center=True, min_periods=1).median()
    return baseline, signal - baseline


# ---------------------------------------------------------------------------
# 信号 vs 真值的对齐程度
# ---------------------------------------------------------------------------
def blink_polarity(signal: np.ndarray, gt: np.ndarray) -> float:
    """
    判断"眨眼时信号是变亮还是变暗"，并返回一个符号，使 blinks 得到高分。

    返回 +1：眨眼帧比基线**更亮**（眼睑/皮肤比深色虹膜亮，本例就是这种情况）
    返回 -1：眨眼帧比基线**更暗**（深色眼睑或睫毛遮住明亮的反光时）
    符号由带符号相关系数决定，不写死方向 —— 换个个体/光照仍可适用。
    """
    corr = float(np.corrcoef(signal, gt)[0, 1])
    return 1.0 if corr >= 0 else -1.0


def detection_score(signal: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, float]:
    """把亮度信号转成"越接近 1 越像眨眼"的分数，并返回所用极性。"""
    polarity = blink_polarity(signal, gt)
    # 标准化到 0~1，便于阈值解释（阈值 = 低于基线多少个标准差）
    sd = np.std(signal) or 1.0
    score = polarity * (signal - np.mean(signal)) / sd
    return score, polarity


def signal_quality(signal: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """单个信号的质量指标：帧级 AUC + 与真值的点二列相关。"""
    score, polarity = detection_score(signal, gt)
    return {
        "frame_auc": float(roc_auc_score(gt, score)),
        "correlation": float(np.corrcoef(signal, gt)[0, 1]),
        "polarity": polarity,
    }


# ---------------------------------------------------------------------------
# 阈值扫描 / 事件匹配
# ---------------------------------------------------------------------------
def threshold_sweep(score: np.ndarray, gt: np.ndarray, n_steps: int = 60) -> pd.DataFrame:
    """在整条分数曲线上扫阈值，记录精确率/召回率/F1 与 Jaccard。"""
    lo, hi = np.percentile(score, 1), np.percentile(score, 99)
    rows = []
    for thr in np.linspace(lo, hi, n_steps):
        pred = (score > thr).astype(int)
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        union = tp + fp + fn
        rows.append(
            {
                "threshold": thr,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard": tp / union if union else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "t12_threshold_sweep.csv", index=False)
    return table


def best_threshold(table: pd.DataFrame) -> pd.Series:
    return table.loc[table["f1"].idxmax()]


def plot_threshold_sweep(table: pd.DataFrame, best: pd.Series):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for col, colour in [("precision", "#4c78a8"), ("recall", "#f58518"), ("f1", "#2e7d32")]:
        ax.plot(table["threshold"], table[col], label=col, color=colour, lw=1.8)
    ax.axvline(best["threshold"], color="crimson", ls="--", lw=1.2)
    ax.set_xlabel("Threshold on standardised blink score")
    ax.set_ylabel("Score")
    ax.set_title(
        f"Best frame-level F1 = {best['f1']:.2f} at threshold {best['threshold']:.2f} "
        f"(P={best['precision']:.2f}, R={best['recall']:.2f})"
    )
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    savefig(fig, "fig33_threshold_sweep")
    return fig


def _runs(mask: np.ndarray, min_length: int = 1) -> list:
    """把布尔序列切成连续的区间列表 [(start, end), ...]。"""
    padded = np.concatenate([[0], mask.astype(int), [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_length]


def event_level_scores(
    score: np.ndarray,
    gt: np.ndarray,
    threshold: float,
    min_length: int = 3,
    tolerance: int = 15,
) -> Dict[str, float]:
    """
    事件级评估：一次眨眼算一个事件，而不是一帧算一个样本。

    只要预测区间的中点落在真值事件的中点 ± tolerance 帧内，就算命中。
    tolerance 默认 15 帧（≈0.25 s），因为亮度变化相对眼睑运动必然有滞后。
    """
    pred_runs = _runs(score > threshold, min_length)
    true_events = groundtruth_intervals(gt)
    true_centres = ((true_events["start"] + true_events["end"]) / 2).values
    pred_centres = np.array([(s + e) / 2 for s, e in pred_runs])

    matched_true = set()
    matched_pred = 0
    for centre in pred_centres:
        if len(true_centres) == 0:
            break
        distances = np.abs(true_centres - centre)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= tolerance and nearest not in matched_true:
            matched_true.add(nearest)
            matched_pred += 1

    tp = matched_pred
    fp = len(pred_runs) - tp
    fn = len(true_centres) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_true_events": len(true_centres),
        "n_predicted_events": len(pred_runs),
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": fn,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "tolerance_frames": tolerance,
    }


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def plot_signal_overview(raw: pd.Series, baseline: pd.Series, corrected: pd.Series, fps: float):
    frames = np.arange(len(raw))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(frames, raw, color="#7f7f7f", lw=0.8, label="Eye brightness")
    axes[0].plot(frames, baseline, color="crimson", lw=1.5,
                 label=f"Baseline (rolling median, 3 s)")
    axes[0].set_ylabel("Mean brightness")
    axes[0].set_title("Eye-region brightness still drifts across the clip")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].plot(frames, corrected, color=COLOR_SIGNAL, lw=0.8)
    axes[1].axhline(0, color="grey", ls="--", lw=1)
    axes[1].set_ylabel("Brightness − baseline")
    axes[1].set_title("After detrending, short dips stand out")

    gt = load_groundtruth()
    axes[2].plot(frames, corrected, color=COLOR_SIGNAL, lw=0.8, label="Corrected brightness")
    ax2 = axes[2].twinx()
    ax2.fill_between(frames, 0, gt, step="post", color=COLOR_TRUTH, alpha=0.25,
                     label="Ground-truth blink")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([])
    ax2.grid(False)
    axes[2].set_xlabel("Frame number")
    axes[2].set_ylabel("Brightness − baseline")
    axes[2].set_title("Do the dips coincide with annotated blinks?")
    fig.tight_layout()
    savefig(fig, "fig31_eye_signal_overview")
    return fig


def plot_zoom(corrected: pd.Series, gt: np.ndarray, start: int = 400, stop: int = 700):
    frames = np.arange(len(corrected))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(frames, corrected.values, color=COLOR_SIGNAL, lw=1.4, label="Corrected brightness")
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xlim(start, stop)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Brightness − baseline")
    ax2 = ax.twinx()
    ax2.fill_between(frames, 0, gt, step="post", color=COLOR_TRUTH, alpha=0.22)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([])
    ax2.grid(False)
    ax.set_title(f"Zoom on frames {start}–{stop}: dips vs annotated blinks")
    fig.tight_layout()
    savefig(fig, "fig32_signal_zoom")
    return fig
