"""
config.py
=========
集中管理路径、常量与绘图风格。

设计原则
--------
1. 原始数据（视频、csv、npy）体积很大（原始视频 52 MB，处理后的视频约 300 MB），
   因此本项目**不复制**原始数据，而是通过 SOURCE_DATA_DIR 指向研学课程包中的
   ``ai_session/data`` 目录。可用环境变量 ``AI_SESSION_DATA`` 覆盖该路径。
2. 所有派生结果（逐帧统计量、跟踪视频、图表、表格）都写入本项目自己的
   ``data/``、``figures/``、``outputs/`` 目录，保证可追溯、可复现。
3. 随机种子、交叉验证折数、跟踪参数等"可调常量"全部集中在此，避免散落在
   各个脚本里导致结果无法对齐。
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]

# 课程原始数据目录（ai_session/data）。默认使用同级目录的相对路径。
_DEFAULT_SOURCE = PROJECT_DIR.parent / "ai_session" / "data"
SOURCE_DATA_DIR = Path(os.environ.get("AI_SESSION_DATA", _DEFAULT_SOURCE))

RAW_VIDEO = SOURCE_DATA_DIR / "raw_videos" / "M-006.mp4"
BLINK_EVENTS_CSV = SOURCE_DATA_DIR / "blink_events.csv"
GROUNDTRUTH_NPY = SOURCE_DATA_DIR / "M-006_groundtruth.npy"

DATA_DIR = PROJECT_DIR / "data"
CROPS_DIR = DATA_DIR / "crops"
PROCESSED_DIR = DATA_DIR / "processed"

FIG_DIR = PROJECT_DIR / "figures"
OUT_DIR = PROJECT_DIR / "outputs"
REPORT_DIR = PROJECT_DIR / "reports"

for _d in (DATA_DIR, CROPS_DIR, PROCESSED_DIR, FIG_DIR, OUT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 由本管线生成的中间产物
FRAME_STATS_CSV = OUT_DIR / "frame_brightness_stats.csv"
TRACKED_VIDEO = PROCESSED_DIR / "tracked_eye_crop.mp4"
ROI_STATS_CSV = OUT_DIR / "roi_signal_stats.csv"
SIGNALS_NPZ = DATA_DIR / "roi_signals.npz"

# ---------------------------------------------------------------------------
# 视频管线参数
# ---------------------------------------------------------------------------
# 眼区初始包围框 (x, y, w, h)，帧尺寸 1080 x 1920。
# 原课程使用 cv2.selectROI 交互式框选；这里把当时选中的框固化为常量，
# 这样任何人在任何机器上都能得到完全一致的结果（可复现性的必要条件）。
EYE_BOX = (250, 470, 640, 560)

# 亮度基线窗口（秒）。课程中：全局亮度用 2 s，眼区亮度用 3 s。
GLOBAL_BASELINE_SECONDS = 2.0
EYE_BASELINE_SECONDS = 3.0

# 跟踪器每 N 帧更新一次（课程默认 100）
FRAME_STEP = 100

# 亮度归一化的修正系数裁剪范围
CORRECTION_CLIP = (0.5, 2.0)

# ---------------------------------------------------------------------------
# 机器学习参数
# ---------------------------------------------------------------------------
RANDOM_STATE = 1
N_SPLITS = 5
N_REPEATS = 10

FEATURES = ["closing_duration", "paused_duration", "opening_duration"]
LABELS = ["F", "H"]
LABEL_NAMES = {"F": "Full blink", "H": "Half blink"}

# ---------------------------------------------------------------------------
# 绘图风格
# ---------------------------------------------------------------------------
COLOR_FULL = "tab:blue"
COLOR_HALF = "tab:orange"
COLOR_SIGNAL = "#1f4e79"
COLOR_TRUTH = "#c0504d"
COLOR_ACCENT = "#2e7d32"


def setup_matplotlib() -> None:
    """统一 matplotlib 风格（字体、网格、DPI）。

    在 Jupyter 里保留交互后端，图会直接显示在 notebook 中；
    在命令行脚本里切到 Agg，避免无显示环境报错。
    """
    import sys

    import matplotlib

    in_notebook = "ipykernel" in sys.modules
    if not in_notebook:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def savefig(fig, name: str) -> Path:
    """把图保存到 figures/ 并返回路径。"""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
