"""
extension_signal.py
===================
延伸分析（二）：把"亮度信号看起来像眨眼"变成"亮度信号能检测多少个眨眼事件"，
并系统比较不同的信号构造方式。

解决的问题（对应报告"延伸"章节）
---------------------------------
S1 该用哪种图像统计量？（均值 / 中位数 / 分位数 / 纹理 / 梯度）
S2 去趋势窗口该多长？（扫描 + 用 AUC 选，而不是用眼睛挑）
S3 去趋势方法该用哪种？（滚动中位数 / Savitzky-Golay / Butterworth 高通 / 一阶差分）
S4 平滑有用吗？（降低噪声 → 事件级精确率提升）
S5 阈值取多少？（帧级 PR 曲线 + 事件级 PR 曲线，两种口径分别给最优阈值）
S6 跟踪每 N 帧更新一次的代价有多大？（frame_step 敏感性）
S7 ROI 真的必要吗？（全帧亮度 vs 眼区亮度）
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, savgol_filter, sosfiltfilt

from config import FRAME_STATS_CSV, OUT_DIR, savefig
import blink_signal as bs

FPS = 59.93488876831253


# ---------------------------------------------------------------------------
# S1 哪种统计量最好
# ---------------------------------------------------------------------------
STAT_LABELS = {
    "roi_mean": "mean brightness",
    "roi_median": "median brightness",
    "roi_std": "contrast (std)",
    "roi_p10": "dark 10th pct",
    "roi_p90": "bright 90th pct",
    "roi_grad_energy": "edge/gradient energy",
    "roi_upper_mean": "upper-half mean (lid)",
    "roi_lower_mean": "lower-half mean",
}


def signal_quality_table(signals: Dict[str, np.ndarray], gt: np.ndarray,
                         window_seconds: float = 3.0) -> pd.DataFrame:
    rows = []
    for key, label in STAT_LABELS.items():
        series = pd.Series(signals[key])
        _, corrected = bs.detrend(series, FPS, window_seconds)
        quality = bs.signal_quality(corrected.to_numpy(), gt)
        rows.append(
            {
                "statistic": label,
                "key": key,
                "frame_auc": quality["frame_auc"],
                "correlation": quality["correlation"],
            }
        )
    table = pd.DataFrame(rows).sort_values("frame_auc", ascending=False).round(3)
    table.to_csv(OUT_DIR / "s1_statistic_quality.csv", index=False)
    return table


def plot_signal_quality(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    colours = ["#2e7d32" if auc > 0.6 else ("#4c78a8" if auc > 0.55 else "#b0b0b0")
               for auc in table["frame_auc"]]
    ax.barh(table["statistic"], table["frame_auc"], color=colours)
    ax.axvline(0.5, color="black", ls="--", lw=1, label="chance")
    for i, auc in enumerate(table["frame_auc"]):
        ax.text(auc + 0.004, i, f"{auc:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.4, 0.78)
    ax.invert_yaxis()
    ax.set_xlabel("Frame-level ROC-AUC against annotated blinks")
    ax.set_title("Mean/median brightness beat contrast and edge energy")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig(fig, "fig50_statistic_quality")
    return fig


# ---------------------------------------------------------------------------
# S2 窗口扫描
# ---------------------------------------------------------------------------
def window_sweep(signals: Dict[str, np.ndarray], gt: np.ndarray,
                 windows: Iterable[float] = (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)) -> pd.DataFrame:
    rows = []
    for key in ("roi_mean", "roi_median", "roi_grad_energy"):
        series = pd.Series(signals[key])
        for window in windows:
            _, corrected = bs.detrend(series, FPS, window)
            quality = bs.signal_quality(corrected.to_numpy(), gt)
            rows.append({"statistic": STAT_LABELS[key], "window_seconds": window,
                         "frame_auc": quality["frame_auc"]})
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "s2_window_sweep.csv", index=False)
    return table


def plot_window_sweep(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for (label, sub), colour in zip(table.groupby("statistic"),
                                    ["#4c78a8", "#f58518", "#2e7d32"]):
        ax.plot(sub["window_seconds"], sub["frame_auc"], "o-", label=label, color=colour)
        best = sub.loc[sub["frame_auc"].idxmax()]
        ax.scatter([best["window_seconds"]], [best["frame_auc"]], s=110, facecolor="none",
                   edgecolor=colour, lw=2, zorder=5)
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.axvspan(1.5, 3.0, color="grey", alpha=0.12)
    ax.set_xlabel("Detrending window (s)")
    ax.set_ylabel("Frame-level ROC-AUC")
    ax.set_title("Window choice matters ~0.07 AUC; 1.5–2 s is near-optimal for all three")
    ax.legend(fontsize=8.5, loc="lower center")
    fig.tight_layout()
    savefig(fig, "fig51_window_sweep")
    return fig


# ---------------------------------------------------------------------------
# S3 去趋势方法比较
# ---------------------------------------------------------------------------
def detrend_rolling(series: pd.Series, window_seconds: float) -> np.ndarray:
    _, corrected = bs.detrend(series, FPS, window_seconds)
    return corrected.to_numpy()


def detrend_savgol(series: pd.Series, window_seconds: float, polyorder: int = 3) -> np.ndarray:
    window = max(5, int(FPS * window_seconds))
    if window % 2 == 0:
        window += 1
    trend = savgol_filter(series.to_numpy(), window_length=window, polyorder=polyorder)
    return series.to_numpy() - trend


def detrend_butter(series: pd.Series, cutoff_hz: float = 0.35, order: int = 3) -> np.ndarray:
    sos = butter(order, cutoff_hz, btype="highpass", fs=FPS, output="sos")
    return sosfiltfilt(sos, series.to_numpy())


def detrend_diff(series: pd.Series, lag: int = 1) -> np.ndarray:
    values = series.to_numpy()
    return np.concatenate([np.zeros(lag), values[lag:] - values[:-lag]])


def detrend_method_comparison(signals: Dict[str, np.ndarray], gt: np.ndarray) -> pd.DataFrame:
    series = pd.Series(signals["roi_mean"])
    variants = {
        "rolling median 2 s": detrend_rolling(series, 2.0),
        "rolling median 3 s (course)": detrend_rolling(series, 3.0),
        "Savitzky-Golay 3 s": detrend_savgol(series, 3.0),
        "Butterworth high-pass 0.35 Hz": detrend_butter(series),
        "first difference (lag 1)": detrend_diff(series, 1),
        "no detrending": series.to_numpy(),
    }
    rows = []
    for name, values in variants.items():
        quality = bs.signal_quality(values, gt)
        score = bs.detection_score(values, gt)[0]
        sweep = bs.threshold_sweep(score, gt)
        best = bs.best_threshold(sweep)
        rows.append(
            {
                "method": name,
                "frame_auc": quality["frame_auc"],
                "best_frame_f1": best["f1"],
                "best_threshold": best["threshold"],
            }
        )
    table = pd.DataFrame(rows).round(3)
    table.to_csv(OUT_DIR / "s3_detrend_methods.csv", index=False)
    return table


def plot_detrend_comparison(signals: Dict[str, np.ndarray], gt: np.ndarray,
                            start: int = 400, stop: int = 620):
    """同一段视频用三种方法去趋势，直观看到"趋势残留"与"过冲"。"""
    series = pd.Series(signals["roi_mean"])
    variants = {
        "rolling median 3 s": detrend_rolling(series, 3.0),
        "Savitzky-Golay 3 s": detrend_savgol(series, 3.0),
        "Butterworth 0.35 Hz": detrend_butter(series),
    }
    frames = np.arange(len(series))
    fig, axes = plt.subplots(len(variants), 1, figsize=(10, 2.4 * len(variants)), sharex=True)
    for ax, (name, values) in zip(axes, variants.items()):
        ax.plot(frames, values, lw=0.9, color="#1f4e79")
        ax.fill_between(frames, values.min(), values.max(), where=gt.astype(bool),
                        color="#c0504d", alpha=0.16, step="post")
        ax.axhline(0, color="grey", ls="--", lw=0.9)
        ax.set_xlim(start, stop)
        ax.set_ylabel(name, fontsize=8)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("Frame number")
    fig.suptitle("Detrending variants on the same window (red = annotated blinks)", y=0.995)
    fig.tight_layout()
    savefig(fig, "fig52_detrend_comparison")
    return fig


# ---------------------------------------------------------------------------
# S4/S5 平滑 + 事件级评估
# ---------------------------------------------------------------------------
def smooth_score(score: np.ndarray, seconds: float = 0.1) -> np.ndarray:
    size = max(1, int(round(FPS * seconds)))
    return uniform_filter1d(score.astype(float), size=size, mode="nearest")


def event_pr_curve(score: np.ndarray, gt: np.ndarray, n_steps: int = 50,
                   min_length: int = 3, tolerance: int = 15) -> pd.DataFrame:
    lo, hi = np.percentile(score, 1), np.percentile(score, 99.5)
    rows = []
    for thr in np.linspace(lo, hi, n_steps):
        stats = bs.event_level_scores(score, gt, thr, min_length=min_length,
                                      tolerance=tolerance)
        rows.append({"threshold": thr, **stats})
    return pd.DataFrame(rows)


def smoothing_ablation(score: np.ndarray, gt: np.ndarray) -> pd.DataFrame:
    """
    平滑窗口 → 事件级最优 F1。这一步回答"课程里没做的一件小事值不值"。
    """
    rows = []
    for seconds in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        smoothed = score if seconds == 0 else smooth_score(score, seconds)
        curve = event_pr_curve(smoothed, gt)
        best = curve.loc[curve["event_f1"].idxmax()]
        rows.append(
            {
                "smoothing_seconds": seconds,
                "best_event_f1": best["event_f1"],
                "event_precision": best["event_precision"],
                "event_recall": best["event_recall"],
                "n_predicted_events": best["n_predicted_events"],
            }
        )
    table = pd.DataFrame(rows).round(3)
    table.to_csv(OUT_DIR / "s4_smoothing_ablation.csv", index=False)
    return table


def plot_event_pr(curve_raw: pd.DataFrame, curve_smooth: pd.DataFrame, label_raw="raw",
                  label_smooth="smoothed 0.1 s"):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(curve_raw["event_recall"], curve_raw["event_precision"], "-o", ms=3,
            color="#7f7f7f", label=label_raw)
    ax.plot(curve_smooth["event_recall"], curve_smooth["event_precision"], "-o", ms=3,
            color="#2e7d32", label=label_smooth)
    for curve, colour in [(curve_raw, "#7f7f7f"), (curve_smooth, "#2e7d32")]:
        best = curve.loc[curve["event_f1"].idxmax()]
        ax.scatter([best["event_recall"]], [best["event_precision"]], s=110,
                   facecolor="none", edgecolor=colour, lw=2, zorder=5)
        ax.annotate(f"F1={best['event_f1']:.2f}", (best["event_recall"],
                    best["event_precision"]), textcoords="offset points",
                    xytext=(8, -12), fontsize=8, color=colour)
    ax.set_xlabel("Event-level recall")
    ax.set_ylabel("Event-level precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Event-level PR: the best threshold sits far above the frame-level optimum")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    savefig(fig, "fig54_event_pr_curve")
    return fig


def plot_smoothed_detection(score: np.ndarray, gt: np.ndarray, threshold: float,
                            start: int, stop: int):
    frames = np.arange(len(score))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True)
    axes[0].plot(frames, score, lw=0.8, color="#7f7f7f", label="frame score")
    axes[0].axhline(threshold, color="crimson", ls="--", lw=1.2, label="threshold")
    axes[0].set_ylabel("Blink score")
    axes[0].set_title("Raw score: short crossings above the threshold inflate the event count")
    axes[0].legend(fontsize=8, loc="upper right")

    smoothed = smooth_score(score, 0.1)
    axes[1].plot(frames, smoothed, lw=1.2, color="#2e7d32", label="smoothed 0.1 s")
    axes[1].axhline(threshold, color="crimson", ls="--", lw=1.2)
    axes[1].fill_between(frames, smoothed.min(), smoothed.max(), where=gt.astype(bool),
                         color="#c0504d", alpha=0.18, step="post", label="annotated blink")
    axes[1].set_xlim(start, stop)
    axes[1].set_xlabel("Frame number")
    axes[1].set_ylabel("Blink score")
    axes[1].set_title("Smoothed score: crossings are cleaner, but the best event F1 is unchanged")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    savefig(fig, "fig55_smoothed_detection")
    return fig


def threshold_criterion_comparison(score: np.ndarray, gt: np.ndarray) -> pd.DataFrame:
    """
    同一个信号，两种"最优阈值"口径：

      A. 按帧级 F1 选阈值（把每一帧当独立样本，眨眼帧只占 20.7%）
      B. 按事件级 F1 选阈值（一次眨眼算一个事件）

    两者差异非常大，说明"先用帧级指标调阈值，再报事件级效果"会系统性低估
    该方法的实际可用性 —— 这是课程里没有涉及、但实际工程必须处理的问题。
    """
    frame_curve = bs.threshold_sweep(score, gt)
    event_curve = event_pr_curve(score, gt)
    frame_best = frame_curve.loc[frame_curve["f1"].idxmax()]
    event_best = event_curve.loc[event_curve["event_f1"].idxmax()]

    at_frame_best = bs.event_level_scores(score, gt, frame_best["threshold"])
    rows = [
        {
            "criterion": "A. frame-level F1 maximum",
            "threshold": frame_best["threshold"],
            "frame_f1": frame_best["f1"],
            "event_f1": at_frame_best["event_f1"],
            "event_precision": at_frame_best["event_precision"],
            "event_recall": at_frame_best["event_recall"],
            "n_predicted_events": at_frame_best["n_predicted_events"],
        },
        {
            "criterion": "B. event-level F1 maximum",
            "threshold": event_best["threshold"],
            "frame_f1": bs.threshold_sweep(score, gt, n_steps=200)
            .iloc[(np.abs(np.linspace(np.percentile(score, 1),
                                      np.percentile(score, 99), 200)
                          - event_best["threshold"]).argmin())]["f1"],
            "event_f1": event_best["event_f1"],
            "event_precision": event_best["event_precision"],
            "event_recall": event_best["event_recall"],
            "n_predicted_events": event_best["n_predicted_events"],
        },
    ]
    table = pd.DataFrame(rows).round(3)
    table.to_csv(OUT_DIR / "s7_threshold_criterion.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# S6 跟踪更新频率的代价
# ---------------------------------------------------------------------------
def tracking_step_robustness(gt: np.ndarray, steps: Iterable[int] = (1, 10, 50, 100)):
    """
    重跑跟踪，只改 frame_step。指标：
      frame_auc      —— 眼区亮度信号的质量
      mean_abs_drift —— 框中心相对初始位置的平移量（像素）

    注意三次运行都用同一套亮度归一化系数，保证唯一的变量是 frame_step。
    """
    import pipeline
    import video

    stats = pipeline.build_frame_stats()
    fps = video.video_info()["fps"]
    _, factors, _ = pipeline.build_global_baseline(stats, fps)

    rows = []
    for step in steps:
        table = video.track_eye(frame_step=step, factors=factors, out_video=None)
        cx = table["x"] + table["w"] / 2
        cy = table["y"] + table["h"] / 2
        drift = float(np.mean(np.hypot(cx - cx.iloc[0], cy - cy.iloc[0])))
        series = table["roi_mean"]
        _, corrected = bs.detrend(series, FPS, 3.0)
        quality = bs.signal_quality(corrected.to_numpy(), gt)
        rows.append(
            {
                "frame_step": step,
                "tracker_updates": int(np.ceil(len(table) / step)),
                "mean_abs_drift_px": drift,
                "frame_auc": quality["frame_auc"],
            }
        )
    table = pd.DataFrame(rows).round(3)
    table.to_csv(OUT_DIR / "s5_tracking_step.csv", index=False)
    return table


def plot_tracking_step(table: pd.DataFrame):
    """
    用等间距的分类轴，而不是对数轴 —— 只有 4 个取值，对数轴会让连线产生误导性的斜率。
    """
    positions = np.arange(len(table))
    fig, ax1 = plt.subplots(figsize=(7.6, 4.2))
    ax1.plot(positions, table["frame_auc"], "o-", color="#2e7d32", label="frame AUC",
             ms=8)
    ax1.set_xticks(positions, [f"every {int(s)} frame(s)" for s in table["frame_step"]])
    ax1.set_xlabel("Tracker update interval")
    ax1.set_ylabel("Frame-level ROC-AUC", color="#2e7d32")
    ax1.tick_params(axis="y", labelcolor="#2e7d32")
    ax1.set_ylim(0.55, 0.70)

    ax2 = ax1.twinx()
    ax2.plot(positions, table["mean_abs_drift_px"], "s--", color="#c0504d",
             label="mean drift", ms=8)
    ax2.set_ylabel("Mean box drift from start (px)", color="#c0504d")
    ax2.tick_params(axis="y", labelcolor="#c0504d")
    ax2.grid(False)

    # 标出最差的一组
    worst = int(np.argmax(table["mean_abs_drift_px"].to_numpy()))
    ax2.annotate("appearance model\ncontaminated by blinks",
                 xy=(positions[worst], table["mean_abs_drift_px"].iloc[worst]),
                 xytext=(18, -6), textcoords="offset points", fontsize=8.5,
                 color="#c0504d",
                 arrowprops=dict(arrowstyle="->", color="#c0504d", lw=1))
    ax1.set_title("Updating every frame is worst: the box drifts most and the signal degrades")
    fig.tight_layout()
    savefig(fig, "fig56_tracking_step")
    return fig


# ---------------------------------------------------------------------------
# S7 ROI vs 全帧
# ---------------------------------------------------------------------------
def global_vs_roi(signals: Dict[str, np.ndarray], gt: np.ndarray) -> pd.DataFrame:
    stats = pd.read_csv(FRAME_STATS_CSV)
    global_series = pd.Series(stats["global_mean"])
    _, global_corrected = bs.detrend(global_series, FPS, 2.0)
    _, roi_corrected = bs.detrend(pd.Series(signals["roi_mean"]), FPS, 3.0)

    rows = []
    for name, values in [("whole-frame brightness", global_corrected.to_numpy()),
                         ("tracked eye region", roi_corrected.to_numpy())]:
        quality = bs.signal_quality(values, gt)
        score = bs.detection_score(values, gt)[0]
        best = bs.best_threshold(bs.threshold_sweep(score, gt))
        rows.append({"signal": name, "frame_auc": quality["frame_auc"],
                     "best_frame_f1": best["f1"]})
    table = pd.DataFrame(rows).round(3)
    table.to_csv(OUT_DIR / "s6_global_vs_roi.csv", index=False)
    return table


def plot_global_vs_roi(signals: Dict[str, np.ndarray], gt: np.ndarray,
                       start: int = 400, stop: int = 700):
    stats = pd.read_csv(FRAME_STATS_CSV)
    _, global_corrected = bs.detrend(pd.Series(stats["global_mean"]), FPS, 2.0)
    _, roi_corrected = bs.detrend(pd.Series(signals["roi_mean"]), FPS, 3.0)
    frames = np.arange(len(global_corrected))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.4), sharex=True)
    for ax, values, title, colour in [
        (axes[0], global_corrected.to_numpy(), "Whole frame (no cropping)", "#7f7f7f"),
        (axes[1], roi_corrected.to_numpy(), "Tracked eye region", "#1f4e79"),
    ]:
        ax.plot(frames, values, lw=0.9, color=colour)
        ax.fill_between(frames, values.min(), values.max(), where=gt.astype(bool),
                        color="#c0504d", alpha=0.16, step="post")
        ax.set_xlim(start, stop)
        ax.set_ylabel("Brightness − baseline")
        ax.set_title(title, fontsize=10)
    axes[-1].set_xlabel("Frame number")
    fig.suptitle("Why crop at all? The eye region tracks the annotated events far better",
                 y=0.995)
    fig.tight_layout()
    savefig(fig, "fig57_global_vs_roi")
    return fig
