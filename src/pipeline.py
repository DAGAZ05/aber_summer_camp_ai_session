"""
pipeline.py
===========
把 Session 2 / Session 3 的步骤串成一条可重复执行的管线，并把中间结果缓存到磁盘。

用法（命令行）::

    python -m pipeline            # 缺什么算什么（有缓存就复用）
    python -m pipeline --force    # 全部重算（约 2 分钟）

缓存文件：
    outputs/frame_brightness_stats.csv  逐帧全局亮度（第一遍遍历视频）
    outputs/roi_signal_stats.csv        逐帧跟踪框 + 眼区统计量（第二遍）
    data/processed/tracked_eye_crop.mp4 跟踪裁出的眼区视频（课程里的产物）
    data/roi_signals.npz                整理好的信号矩阵 + 真值
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    EYE_BASELINE_SECONDS,
    FRAME_STEP,
    FRAME_STATS_CSV,
    GLOBAL_BASELINE_SECONDS,
    RAW_VIDEO,
    ROI_STATS_CSV,
    SIGNALS_NPZ,
    TRACKED_VIDEO,
)
import video


def build_frame_stats(force: bool = False) -> pd.DataFrame:
    if force and FRAME_STATS_CSV.exists():
        FRAME_STATS_CSV.unlink()
    return video.compute_frame_brightness(cache=not force)


def build_global_baseline(stats: pd.DataFrame, fps: float):
    baseline = video.rolling_baseline(stats["global_mean"], fps, GLOBAL_BASELINE_SECONDS)
    factors, reference = video.correction_factors(baseline)
    return baseline, factors, reference


def build_roi_table(force: bool = False) -> pd.DataFrame:
    """第二遍遍历：边归一化边跟踪眼区，输出逐帧统计量与眼区裁片视频。"""
    if ROI_STATS_CSV.exists() and not force:
        return pd.read_csv(ROI_STATS_CSV)

    stats = build_frame_stats(force=False)
    fps = video.video_info()["fps"]
    _, factors, _ = build_global_baseline(stats, fps)

    table = video.track_eye(
        frame_step=FRAME_STEP,
        factors=factors,
        out_video=TRACKED_VIDEO,
    )
    table.attrs = {}
    table.to_csv(ROI_STATS_CSV, index=False)
    return table


def build_signals(force: bool = False) -> dict:
    """把 ROI 统计量 + 真值打包成 npz，供笔记本快速加载。"""
    if SIGNALS_NPZ.exists() and not force:
        data = np.load(SIGNALS_NPZ)
        return {k: data[k] for k in data.files}

    roi = build_roi_table(force=force)
    gt = np.load(RAW_VIDEO.parents[1] / "M-006_groundtruth.npy").astype(int).ravel()
    columns = [c for c in roi.columns if c.startswith("roi_")]
    payload = {c: roi[c].to_numpy() for c in columns}
    payload["groundtruth"] = gt
    payload["frame"] = roi["frame"].to_numpy()
    payload["box_x"] = roi["x"].to_numpy()
    payload["box_y"] = roi["y"].to_numpy()
    np.savez_compressed(SIGNALS_NPZ, **payload)
    return payload


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    force = "--force" in argv

    t0 = time.time()
    info = video.video_info()
    print(f"Video: {Path(info['file']).name} | frames={info['frame_count']} | "
          f"fps={info['fps']:.2f} | {info['width']}x{info['height']} | "
          f"duration={info['duration_seconds']:.2f} s")

    stats = build_frame_stats(force=force)
    baseline, factors, reference = build_global_baseline(stats, info["fps"])
    print(f"Global brightness: mean={stats['global_mean'].mean():.1f} | "
          f"baseline reference={reference:.2f} | "
          f"correction factors {factors.min():.2f}–{factors.max():.2f}")

    roi = build_roi_table(force=force)
    print(f"ROI table: {len(roi)} frames | tracker={video.make_tracker()[1]} | "
          f"frame_step={FRAME_STEP}")

    payload = build_signals(force=force)
    print(f"Saved signals: {list(payload)}")
    print(f"Done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
