"""
build_notebooks.py
==================
把 `notebooks_src/*.py`（带 `# %%` 单元标记的"文学化脚本"）转换成可执行的
`.ipynb`，并（可选）执行一遍把输出嵌进去。

为什么用这种方式而不是手写 .ipynb：
  * .py 版本可以 diff、可以 lint、可以当脚本直接跑；
  * 需要交付 notebook 时一条命令重新生成，避免"notebook 与代码不同步"；
  * 执行一遍后 notebook 里带着全部输出（图、表、数字），可以直接给人看。

用法::

    python tools/build_notebooks.py             # 只转换
    python tools/build_notebooks.py --execute   # 转换 + 执行（输出嵌入）
    python tools/build_notebooks.py --execute 01_session1_ml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import nbformat as nbf

PROJECT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT / "notebooks_src"
OUT_DIR = PROJECT / "notebooks"

MARKER = re.compile(r"^#\s*%%(?:\s*\[markdown\])?\s*$")


def parse_cells(text: str) -> list:
    """按 `# %%` / `# %% [markdown]` 切分单元。"""
    cells, buffer, kind = [], [], "code"
    for line in text.splitlines():
        if MARKER.match(line):
            if buffer:
                cells.append((kind, "\n".join(buffer).strip("\n")))
            kind = "markdown" if "[markdown]" in line else "code"
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        cells.append((kind, "\n".join(buffer).strip("\n")))
    return [(k, c) for k, c in cells if c.strip()]


def to_markdown(text: str) -> str:
    """去掉注释前缀，保留缩进与代码块。"""
    lines = []
    inside_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            inside_fence = not inside_fence
            lines.append(line)
            continue
        if inside_fence:
            lines.append(line)
            continue
        if line.startswith("# "):
            lines.append(line[2:])
        elif line.strip() == "#":
            lines.append("")
        elif line.startswith("#"):
            lines.append(line[1:])
        else:
            lines.append(line)
    return "\n".join(lines)


def build_one(path: Path) -> Path:
    cells = parse_cells(path.read_text(encoding="utf-8"))
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3 (ai-session)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        }
    )
    for kind, content in cells:
        if kind == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(to_markdown(content)))
        else:
            nb.cells.append(nbf.v4.new_code_cell(content))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}.ipynb"
    nbf.write(nb, out)
    return out


def execute_one(path: Path, timeout: int = 1800) -> None:
    from nbclient import NotebookClient

    nb = nbf.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT)}},
        allow_errors=False,
    )
    client.execute()
    nbf.write(nb, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="只处理指定的 notebook 名（不含后缀）")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    sources = sorted(SRC_DIR.glob("*.py"))
    if args.names:
        sources = [p for p in sources if p.stem in set(args.names)]

    for src in sources:
        out = build_one(src)
        print(f"[build] {src.name} -> {out.relative_to(PROJECT)}")
        if args.execute:
            execute_one(out, timeout=args.timeout)
            print(f"[exec ] {out.name} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
