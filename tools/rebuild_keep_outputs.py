"""
rebuild_keep_outputs.py
=======================
改完 `notebooks_src/*.py` 的**文字**（markdown）后，不必重跑几十分钟：
本脚本重新生成 notebook，并把旧 notebook 中**代码单元内容未变**的输出原样搬过来。

- 代码单元内容变了 → 输出丢弃（需要重新执行，避免"代码与输出不一致"）
- 只有 markdown 变了 → 输出完整保留

用法::

    python tools/rebuild_keep_outputs.py 04_extension_model_evaluation
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT / "notebooks_src"
OUT_DIR = PROJECT / "notebooks"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebooks import build_one  # noqa: E402


def rebuild(stem: str) -> None:
    old_path = OUT_DIR / f"{stem}.ipynb"
    old = nbf.read(old_path, as_version=4) if old_path.exists() else None

    new_path = build_one(SRC_DIR / f"{stem}.py")
    new = nbf.read(new_path, as_version=4)

    if old is None:
        print(f"[skip] {stem}: 没有旧 notebook，需完整执行")
        return

    old_code = [c for c in old.cells if c.cell_type == "code"]
    kept = dropped = 0
    for cell in new.cells:
        if cell.cell_type != "code":
            continue
        for candidate in old_code:
            if candidate.source == cell.source:
                cell.outputs = candidate.outputs
                cell.execution_count = candidate.execution_count
                kept += 1
                break
        else:
            dropped += 1

    nbf.write(new, new_path)
    print(f"[rebuild] {stem}: 保留输出 {kept} 个单元，丢弃 {dropped} 个")
    if dropped:
        print("           ↑ 这些单元需要重新执行才能补上输出")


if __name__ == "__main__":
    stems = sys.argv[1:]
    if not stems:
        stems = [p.stem for p in sorted(SRC_DIR.glob("*.py"))]
    for s in stems:
        rebuild(s)
