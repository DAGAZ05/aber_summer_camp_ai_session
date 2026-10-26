"""
video.py
========
第四部分（对应课程 01-image-analysis.html 与 final session）：把视频变成可分析的
数据。

管线（与课程完全一致的五步）：
  1. 打开视频，读元数据（帧数 / fps / 分辨率 / 时长）
  2. 逐帧转灰度，并记录每帧的全局平均亮度
  3. 用滚动中位数估计"缓慢变化的亮度基线"，把全局亮度去趋势
  4. 用基线把每一帧的亮度归一化（乘一个裁剪过的修正系数）
  5. 在第 1 帧框选眼部 ROI，用 CSRT 跟踪器跟踪，裁出眼区
  6. 从眼区裁片里提取亮度信号

工程上的两点取舍（在报告中说明）：
  * 原课程把灰度视频与归一化视频各存成 100+ MB 的 mp4；本项目**不落盘**这两个
    中间视频，而是在第二遍读取时"边归一化边跟踪"，只保留逐帧统计量和小体积
    的眼区裁片视频。数值结果与课程一致，磁盘占用从 ~300 MB 降到 ~40 MB。
  * 原课程用 cv2.selectROI 交互式框选，结果无法复现。本项目把框固化为
    config.EYE_BOX，任何人都能重跑出同样的数字。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CORRECTION_CLIP,
    EYE_BOX,
    FRAME_STATS_CSV,
    RAW_VIDEO,
    ROI_STATS_CSV,
    savefig,
)


# ---------------------------------------------------------------------------
# 1. 打开视频 / 元数据
# ---------------------------------------------------------------------------
def video_info(path: Path = RAW_VIDEO) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    info = {
        "file": str(path),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_seconds"] = info["frame_count"] / info["fps"]
    cap.release()
    return info


def read_frame(path: Path, frame_number: int) -> np.ndarray:
    """读取指定帧（BGR）。超出范围时抛出明确错误。"""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_number} from {path}")
    return frame


def plot_frame(frame_bgr: np.ndarray, title: str, name: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if name:
        savefig(fig, name)
    return fig


def plot_frame_with_box(frame_bgr: np.ndarray, box: Tuple[int, int, int, int],
                        title: str, name: Optional[str] = None):
    x, y, w, h = box
    vis = frame_bgr.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 5)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    if name:
        savefig(fig, name)
    return fig


# ---------------------------------------------------------------------------
# 2. 灰度 + 逐帧亮度
# ---------------------------------------------------------------------------
def compute_frame_brightness(path: Path = RAW_VIDEO, cache: bool = True) -> pd.DataFrame:
    """
    一次遍历：每帧转灰度，记录全局平均亮度与对比度（标准差）。

    课程里同时把灰度帧像素全部存进内存（all_pixels），对 1841 帧 1080×1920
    的视频约需 3.7 GB 内存；这里改为流式统计，只保留标量，效果等价。
    """
    if cache and FRAME_STATS_CSV.exists():
        return pd.read_csv(FRAME_STATS_CSV)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    means, stds = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(float(gray.mean()))
        stds.append(float(gray.std()))
    cap.release()

    table = pd.DataFrame(
        {
            "frame": np.arange(len(means)),
            "global_mean": means,
            "global_std": stds,
        }
    )
    table.to_csv(FRAME_STATS_CSV, index=False)
    return table


def rolling_baseline(series: pd.Series, fps: float, seconds: float) -> pd.Series:
    """居中滚动中位数作为"缓慢变化的亮度基线"。"""
    window = max(3, int(fps * seconds))
    if window % 2 == 0:
        window += 1
    return series.rolling(window=window, center=True, min_periods=1).median()


def plot_global_brightness(stats: pd.DataFrame, baseline: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(stats["frame"], stats["global_mean"], color="#7f7f7f", lw=1,
                 label="Global mean brightness")
    axes[0].plot(stats["frame"], baseline, color="crimson", lw=1.6,
                 label="Rolling median baseline (2 s)")
    axes[0].set_ylabel("Mean brightness")
    axes[0].set_title("Global brightness drifts with lighting and camera exposure")
    axes[0].legend(fontsize=8, loc="lower right")

    detrended = stats["global_mean"] - baseline
    axes[1].plot(stats["frame"], detrended, color="#1f4e79", lw=0.9)
    axes[1].axhline(0, color="grey", ls="--", lw=1)
    axes[1].set_xlabel("Frame number")
    axes[1].set_ylabel("Brightness − baseline")
    axes[1].set_title("After removing the baseline, only short-term changes remain")
    fig.tight_layout()
    savefig(fig, "fig23_global_brightness")
    return fig


def correction_factors(baseline: pd.Series) -> np.ndarray:
    """
    归一化系数 = 参考亮度 / 当前基线，并裁剪到 [0.5, 2.0] 防止过度修正。
    与课程逐字对应（含除以零保护）。
    """
    reference = float(baseline.median())
    factors = np.ones(len(baseline))
    for i, value in enumerate(baseline.values):
        factors[i] = reference / value if value > 0 else 1.0
    return np.clip(factors, *CORRECTION_CLIP), reference


def normalise_frame(gray: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(gray.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def compare_normalisation(factors: np.ndarray, frames=(0, 600, 1200)):
    """可视化：原始灰度 vs 归一化后，直方图与图像并排。"""
    cap = cv2.VideoCapture(str(RAW_VIDEO))
    fig, axes = plt.subplots(len(frames), 3, figsize=(11, 3.4 * len(frames)))
    for row, frame_number in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm = normalise_frame(gray, factors[frame_number])
        axes[row, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(f"Frame {frame_number}: greyscale", fontsize=9)
        axes[row, 1].imshow(norm, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(
            f"after ×{factors[frame_number]:.2f} normalisation", fontsize=9)
        axes[row, 2].hist(gray.ravel(), bins=64, alpha=0.55, label="raw", color="grey")
        axes[row, 2].hist(norm.ravel(), bins=64, alpha=0.55, label="normalised",
                          color="#1f4e79")
        axes[row, 2].set_title(f"Pixel histogram (mean {gray.mean():.0f} → "
                               f"{norm.mean():.0f})", fontsize=9)
        axes[row, 2].legend(fontsize=7)
        for col in range(3):
            axes[row, col].grid(False)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    cap.release()
    fig.tight_layout()
    savefig(fig, "fig22_normalisation_comparison")
    return fig


# ---------------------------------------------------------------------------
# 3. 跟踪眼区 ROI
# ---------------------------------------------------------------------------
def make_tracker():
    """优先用 CSRT（精度高），回退到 KCF；都没有则报错并提示装 contrib 版。"""
    for name in ("TrackerCSRT_create", "TrackerKCF_create"):
        if hasattr(cv2, name):
            return getattr(cv2, name)(), name
    if hasattr(cv2, "legacy"):
        for name in ("TrackerCSRT_create", "TrackerKCF_create"):
            if hasattr(cv2.legacy, name):
                return getattr(cv2.legacy, name)(), "legacy." + name
    raise AttributeError(
        "No OpenCV tracker available. Install opencv-contrib-python and restart."
    )


def roi_statistics(gray_crop: np.ndarray) -> dict:
    """
    从一张眼区裁片里提取多个统计量（不只是均值）。

    这是延伸分析的基础：均值把"整块变暗"和"局部结构变化"混在一起，
    加入中位数、分位数、纹理（标准差）、梯度能量后，可以比较"哪种描述
    更接近眨眼事件"。
    """
    flat = gray_crop.astype(np.float32)
    gx = cv2.Sobel(flat, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(flat, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    h, w = flat.shape
    return {
        "roi_mean": float(flat.mean()),
        "roi_median": float(np.median(flat)),
        "roi_std": float(flat.std()),
        "roi_p10": float(np.percentile(flat, 10)),
        "roi_p90": float(np.percentile(flat, 90)),
        "roi_grad_energy": float(grad.mean()),
        # 上下两个子区域：上=上眼睑/睫毛，下=角膜下缘+皮肤
        "roi_upper_mean": float(flat[: h // 2].mean()),
        "roi_lower_mean": float(flat[h // 2:].mean()),
    }


def track_eye(
    box: Tuple[int, int, int, int] = EYE_BOX,
    frame_step: int = 100,
    factors: Optional[np.ndarray] = None,
    out_video: Optional[Path] = None,
    max_frames: Optional[int] = None,
) -> pd.DataFrame:
    """
    在（亮度归一化后的）灰度视频上跟踪眼区，返回逐帧表：
        frame, x, y, w, h, tracked, roi_* 统计量

    frame_step 与原课程一致：为了省时间，每 N 帧才真正调用一次 tracker.update，
    中间帧复用上一次的框。这一步的代价会在延伸分析里被量化。
    """
    cap = cv2.VideoCapture(str(RAW_VIDEO))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {RAW_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, first = cap.read()
    if not ok:
        raise ValueError("Could not read the first frame")

    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    if factors is not None:
        first_gray = normalise_frame(first_gray, factors[0])

    tracker, tracker_name = make_tracker()
    tracker.init(first_gray, box)

    x, y, w, h = [int(v) for v in box]
    writer = None
    if out_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

    rows = []
    last_box = (x, y, w, h)
    frame_number = 0
    while True:
        if frame_number == 0:
            gray = first_gray
        else:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if factors is not None:
                gray = normalise_frame(gray, factors[frame_number])

        if frame_number % frame_step == 0:
            success, updated = tracker.update(gray)
            if success:
                last_box = tuple(int(v) for v in updated)
        else:
            success = True

        bx, by, bw, bh = last_box
        H, W = gray.shape
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(W, bx + bw), min(H, by + bh)

        if success and x2 > x1 and y2 > y1:
            crop = cv2.resize(gray[y1:y2, x1:x2], (w, h))
        else:
            crop = np.zeros((h, w), dtype=np.uint8)

        stats = roi_statistics(crop)
        rows.append({"frame": frame_number, "x": bx, "y": by, "w": bw, "h": bh,
                     "tracked": bool(success), **stats})

        if writer is not None:
            writer.write(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))

        frame_number += 1
        if max_frames is not None and frame_number >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    table = pd.DataFrame(rows)
    table.attrs["tracker"] = tracker_name
    table.attrs["frame_step"] = frame_step
    return table


def save_crop_frames(frames, out_dir: Path, box: Tuple[int, int, int, int] = EYE_BOX,
                     prefix: str = "crop") -> list:
    """
    把若干帧的眼区裁片存成 PNG。

    注意：Windows 下 cv2.imwrite 遇到中文路径会静默失败，所以统一用
    imencode + 二进制写入，保证在任何路径下都能落盘。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(RAW_VIDEO))
    x, y, w, h = box
    written = []
    for number in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
        path = out_dir / f"{prefix}_{number:05d}.png"
        ok, buf = cv2.imencode(".png", crop)
        if ok:
            path.write_bytes(buf.tobytes())
            written.append(path)
    cap.release()
    return written


def plot_blink_sequence(frames, box: Tuple[int, int, int, int] = EYE_BOX):
    """
    取一次眨眼前后若干帧，拼成"眼区变化"序列图 —— 直观看到亮度为什么会变。
    同时在上方标出每帧的眼区平均亮度。
    """
    cap = cv2.VideoCapture(str(RAW_VIDEO))
    x, y, w, h = box
    crops, means = [], []
    for number in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
        crops.append(crop)
        means.append(crop.mean())
    cap.release()

    fig, axes = plt.subplots(1, len(crops), figsize=(2.4 * len(crops), 3.0))
    for ax, crop, number, mean in zip(np.atleast_1d(axes), crops, frames, means):
        ax.imshow(crop, cmap="gray")
        ax.set_title(f"f{number}\nmean {mean:.0f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("One blink, frame by frame: the eye region brightens as the lid covers it",
                 y=1.04, fontsize=10)
    fig.tight_layout()
    savefig(fig, "fig26_blink_sequence")
    return fig, list(zip(frames, np.round(means, 1)))


def plot_tracking_drift(table: pd.DataFrame):
    """扩展用：框中心的位移曲线 —— 判断跟踪是否漂移。"""
    cx = table["x"] + table["w"] / 2
    cy = table["y"] + table["h"] / 2
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(table["frame"], cx - cx.iloc[0], label="Δ centre x", lw=1)
    ax.plot(table["frame"], cy - cy.iloc[0], label="Δ centre y", lw=1)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Offset from initial centre (px)")
    ax.set_title("Tracker drift: how far the box wanders from its starting position")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "fig24_tracker_drift")
    return fig


def plot_eye_crop_sequence(table_with_capture=None, frames=(0, 40, 80, 120)):
    """取若干帧的眼区裁片拼成一行，展示眨眼过程中眼区外观的变化。"""
    cap = cv2.VideoCapture(str(RAW_VIDEO))
    crops = []
    for n in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = EYE_BOX
        crops.append(gray[y:y + h, x:x + w])
    cap.release()

    fig, axes = plt.subplots(1, len(crops), figsize=(3 * len(crops), 2.8))
    for ax, crop, n in zip(np.atleast_1d(axes), crops, frames):
        ax.imshow(crop, cmap="gray")
        ax.set_title(f"frame {n}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    savefig(fig, "fig25_eye_crops")
    return fig
