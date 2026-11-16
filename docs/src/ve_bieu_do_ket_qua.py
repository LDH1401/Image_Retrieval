"""Biểu đồ R@10 test của các cấu hình (a)–(g) trên FashionIQ dress.

Số liệu đọc từ output/huong1/huong1_tong_hop.json và
output/huong3/huong3_tong_hop.json; ảnh ghi vào docs/images/08_ket_qua_r10.png.

Chạy từ thư mục magiclens/ (cần matplotlib; ảnh trong repo vẽ bằng 3.11.2):
  python docs/src/ve_bieu_do_ket_qua.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.patches import Patch, Polygon  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
ZERO_SHOT, SUPERVISED = "#2a78d6", "#eb6834"  # hai màu đầu của bảng màu phân loại
OUT = "docs/images/08_ket_qua_r10.png"


def rounded_bar(ax, y, value, height, radius_in, color):
  """Thanh ngang vuông ở gốc, bo tròn ở đầu mút (bán kính tính theo inch)."""
  bbox = ax.get_window_extent()
  dpi = ax.figure.dpi
  x0, x1 = ax.get_xlim()
  y0, y1 = ax.get_ylim()
  rx = radius_in * dpi * (x1 - x0) / bbox.width
  ry = radius_in * dpi * abs(y1 - y0) / bbox.height
  t = np.linspace(0, np.pi / 2, 12)
  bottom, top = y - height / 2, y + height / 2
  pts = [(0, bottom), (value - rx, bottom)]
  pts += [(value - rx + rx * np.sin(a), bottom + ry - ry * np.cos(a)) for a in t]
  pts += [(value - rx + rx * np.cos(a), top - ry + ry * np.sin(a)) for a in t]
  pts += [(value - rx, top), (0, top)]
  ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor="none", zorder=3))


def main():
  with open("output/huong1/huong1_tong_hop.json") as f:
    h1 = json.load(f)["configs"]
  with open("output/huong3/huong3_tong_hop.json") as f:
    h3 = json.load(f)["rows"]
  run = lambda prefix: next(v for k, v in h3.items() if k.startswith(prefix))["test"]["R@10"]
  bars = [("(a) MagicLens zero-shot", h1["a"]["test"]["R@10"], ZERO_SHOT),
          ("(b) (a) + VLM 2B", h1["b"]["test"]["R@10"], ZERO_SHOT),
          ("(c) MagicLens tinh chỉnh", h1["c"]["test"]["R@10"], SUPERVISED),
          ("(d) (c), chỉ dùng VLM 2B", h1["d"]["test"]["R@10"], SUPERVISED),
          ("(e) (c) + VLM 2B", h1["e"]["test"]["R@10"], SUPERVISED),
          ("(f) (c) + VLM 7B", run("MagicLens + Qwen2-VL-7B"), SUPERVISED),
          ("(g) (c) + VLM 2B + LoRA", run("MagicLens + Qwen2-VL-2B bf16 + LoRA"), SUPERVISED)]
  ceiling = h1["c"]["test"]["R@50"]

  plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
  fig, ax = plt.subplots(figsize=(6.3, 3.0), dpi=300)
  fig.patch.set_facecolor(SURFACE)
  ax.set_facecolor(SURFACE)
  ax.set_xlim(0, 62)
  ax.set_ylim(len(bars) - 0.4, -0.6)
  fig.subplots_adjust(left=0.27, right=0.98, top=0.86, bottom=0.15)
  ax.set_yticks(range(len(bars)), [b[0] for b in bars], color=INK)
  ax.tick_params(axis="y", length=0, pad=6)
  ax.set_xticks(range(0, 61, 10))
  ax.tick_params(axis="x", colors=INK_2, length=0, labelsize=8)
  ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
  for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
  ax.spines["bottom"].set_color(GRID)
  ax.set_xlabel("R@10 trên 931 truy vấn test (%)", color=INK_2, fontsize=8)

  for y, (_, value, color) in enumerate(bars):
    rounded_bar(ax, y, value, 0.56, 0.04, color)
    ax.text(value + 0.8, y, f"{value:.1f}", va="center", ha="left", color=INK, zorder=4)

  first, last = 2, len(bars) - 1  # (c)–(g): cùng tập ứng viên top-50 của (c)
  ax.plot([ceiling, ceiling], [first - 0.4, last + 0.4], color=INK_2, linewidth=0.9,
          linestyle=(0, (3, 2)), zorder=2)
  ax.text(ceiling - 0.6, first - 0.5, f"trần khi xếp lại: {ceiling:.1f}",
          ha="right", va="center", color=INK_2, fontsize=7.5)

  ax.legend(handles=[Patch(color=ZERO_SHOT, label="Không học trên FIQ"),
                     Patch(color=SUPERVISED, label="Có học trên tập train FIQ")],
            loc="lower left", bbox_to_anchor=(-0.36, 1.02), ncol=2, frameon=False,
            handlelength=1.2, handleheight=0.9, fontsize=8, labelcolor=INK)
  fig.savefig(OUT, dpi=300, facecolor=SURFACE)
  print("Đã ghi", OUT)


if __name__ == "__main__":
  main()
