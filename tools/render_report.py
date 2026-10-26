"""
render_report.py
================
把 `reports/复现与延伸报告.md` 渲染成**自包含**的 HTML：图片以 base64 内嵌，
单个文件即可发给别人看，不依赖相对路径。

用法::

    python tools/render_report.py
"""

from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

import markdown

PROJECT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT / "reports"

CSS = """
:root { --ink:#1a1a1a; --muted:#5f6b7a; --line:#e3e8ef; --accent:#1f4e79; --bg:#ffffff; }
* { box-sizing: border-box; }
body {
  margin: 0 auto; max-width: 980px; padding: 48px 28px 96px;
  font-family: -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
  color: var(--ink); background: var(--bg); line-height: 1.78; font-size: 15.5px;
}
h1 { font-size: 30px; line-height: 1.3; margin: 0 0 8px; letter-spacing: -0.3px; }
h2 { font-size: 22px; margin: 46px 0 14px; padding-bottom: 8px; border-bottom: 2px solid var(--line); }
h3 { font-size: 17.5px; margin: 30px 0 10px; color: var(--accent); }
h4 { font-size: 15.5px; margin: 22px 0 8px; }
p, li { color: #23282f; }
blockquote {
  margin: 18px 0; padding: 14px 18px; border-left: 4px solid var(--accent);
  background: #f5f8fc; color: #26313d; border-radius: 0 6px 6px 0;
}
blockquote p { margin: 0; }
code { background: #f2f4f7; padding: 2px 6px; border-radius: 4px; font-size: 13.5px;
  font-family: "Cascadia Mono", Consolas, monospace; color: #b03060; }
pre { background: #1f2430; color: #e6e6e6; padding: 14px 16px; border-radius: 8px;
  overflow-x: auto; font-size: 13px; line-height: 1.6; }
pre code { background: none; color: inherit; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14px; }
th, td { border: 1px solid var(--line); padding: 8px 11px; text-align: left; vertical-align: top; }
th { background: #f5f7fa; font-weight: 600; }
tr:nth-child(even) td { background: #fbfcfe; }
img { max-width: 100%; display: block; margin: 18px auto 6px; border-radius: 6px;
  border: 1px solid var(--line); }
em.caption { display: block; text-align: center; color: var(--muted); font-size: 13px;
  margin-bottom: 26px; font-style: normal; }
hr { border: none; border-top: 1px solid var(--line); margin: 40px 0; }
.meta { color: var(--muted); font-size: 13.5px; margin-bottom: 28px; }
ul, ol { padding-left: 24px; }
li { margin: 4px 0; }
strong { color: #11161c; }
"""


def embed_images(html: str, base: Path) -> str:
    """把 <img src="..."> 换成 data URI。"""

    def replace(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith(("data:", "http")):
            return match.group(0)
        path = (base / src).resolve()
        if not path.exists():
            return match.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', replace, html)


def number_figures(html: str) -> str:
    """给每个 <img> 后面补一个"图 n"说明，方便正文引用。"""
    counter = {"n": 0}

    def replace(match: re.Match) -> str:
        counter["n"] += 1
        return f'{match.group(0)}<em class="caption">图 {counter["n"]}</em>'

    return re.sub(r"<img[^>]*>", replace, html)


def main() -> int:
    md_path = REPORT_DIR / "复现与延伸报告.md"
    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
    )
    html_body = embed_images(html_body, REPORT_DIR)
    html_body = number_figures(html_body)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>眨眼检测：课程复现与延伸分析</title>
<style>{CSS}</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    out = REPORT_DIR / "复现与延伸报告.html"
    out.write_text(html, encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"[report] {out.relative_to(PROJECT)}  ({size_mb:.1f} MB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
