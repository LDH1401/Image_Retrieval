"""Vẽ các sơ đồ kiến trúc pipeline MagicLens.

Cách chạy (cần matplotlib):
    python docs/src/ve_so_do.py docs/images
"""
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/images"
os.makedirs(OUT, exist_ok=True)

# Bảng màu
INK, INK2, MUTED, LINE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
C = {
    "data": ("#e3eefb", "#2a78d6"),
    "train": ("#fde8dd", "#eb6834"),
    "infer": ("#dcf3ea", "#1baf7a"),
    "model": ("#ebe8fa", "#6250d6"),
    "neutral": ("#f5f4f0", "#c3c2b7"),
    "white": ("#ffffff", "#c3c2b7"),
    "warn": ("#fff4dc", "#eda100"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
PT = 1 / 7.2  # 1 pt tính theo đơn vị toạ độ (1 inch = 10 đơn vị)


def canvas(w_in, h_in):
    fig = plt.figure(figsize=(w_in, h_in), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_in * 10)
    ax.set_ylim(0, h_in * 10)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title=None, lines=(), kind="neutral", fs=8.5, title_fs=None,
        align="left", bullet=True, lw=1.2, title_color=None, va_center=False):
    """Hộp bo góc; (x, y) là góc dưới-trái. Trả về toạ độ các cạnh để nối mũi tên."""
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    title_fs = title_fs or fs + 1
    lh = fs * 1.38 * PT
    char_w = fs * 0.56 * PT
    width_chars = max(8, int((w - 2.4) / char_w))
    body = []
    for ln in lines:
        prefix = "• " if bullet and ln and not ln.startswith(("→", " ")) else ""
        wrapped = textwrap.wrap(ln, width=width_chars - len(prefix)) or [""]
        body.append(prefix + wrapped[0])
        body += [("  " if prefix else "") + t for t in wrapped[1:]]
    n_title = 0
    title_lines = []
    if title:
        title_lines = textwrap.wrap(title, width=int((w - 2) / (title_fs * 0.6 * PT)))
        n_title = len(title_lines)
    total_h = n_title * title_fs * 1.35 * PT + (0.8 if title and body else 0) + len(body) * lh
    if va_center:
        cy = y + h / 2 + total_h / 2
    else:
        cy = y + h - 1.6
    for tl in title_lines:
        ax.text(x + w / 2, cy, tl, ha="center", va="top", fontsize=title_fs, fontweight="bold",
                color=title_color or INK, zorder=3)
        cy -= title_fs * 1.35 * PT
    if title and body:
        cy -= 0.8
    tx = x + 1.2 if align == "left" else x + w / 2
    for ln in body:
        ax.text(tx, cy, ln, ha="left" if align == "left" else "center", va="top", fontsize=fs,
                color=INK2 if title else INK, zorder=3)
        cy -= lh
    return dict(l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h), b=(x + w / 2, y))


def arrow(ax, p1, p2, label=None, color=INK2, rad=0.0, lw=1.4, fs=8, label_dy=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4, shrinkA=2, shrinkB=2))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + label_dy, label, ha="center", va="bottom", fontsize=fs, color=INK2, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))


def poly_arrow(ax, pts, color=INK2, lw=1.4):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=4, solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def badge(ax, x, y, text, kind):
    fc, ec = C[kind]
    ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=ec, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc="white", ec=ec, lw=1))


def title(ax, x, y, text, sub=None):
    ax.text(x, y, text, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    if sub:
        ax.text(x, y - 3.0, sub, ha="left", va="top", fontsize=9, color=MUTED)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("saved", name)


# =====================================================================
# 1. Tổng quan
# =====================================================================
fig, ax = canvas(15, 4.5)
title(ax, 2, 43, "Pipeline MagicLens: tổng quan ba giai đoạn",
      "A và B được thực hiện bởi tác giả (không có trong repo); C là phần có thể chạy với mã nguồn công bố")
bw, bh, by = 41, 25, 7.5
A = box(ax, 2, by, bw, bh, "A. XÂY DỰNG DỮ LIỆU", [
    "Nguồn: Common Crawl, gom ảnh theo URL trang web",
    "Làm sạch: khử trùng lặp, lọc độ phân giải, quảng cáo",
    "Siêu dữ liệu: alt-text, nhãn ICA (Vision API), chú thích PaLI",
    "Lọc cặp ảnh theo độ liên quan thị giác / phi thị giác",
    "PaLM 2 viết chỉ dẫn nối ảnh truy vấn với ảnh đích",
    "→ Kết quả: 36.7 triệu bộ ba (ảnh truy vấn, chỉ dẫn, ảnh đích)",
], kind="data", fs=9, title_fs=11, title_color=C["data"][1])
B = box(ax, 54, by, bw, bh, "B. HUẤN LUYỆN", [
    "Mô hình E dùng chung tham số cho truy vấn và ảnh đích",
    "Backbone CLIP hoặc CoCa + 4 khối self-attention + attention pooling",
    "Loss tương phản: ảnh đích khác trong batch + mẫu âm là chính ảnh truy vấn",
    "Batch 2048, 50 000 bước (~2.8 epoch), Adafactor",
    "64 / 128 TPU, khoảng 6 giờ",
    "→ Kết quả: checkpoint MagicLens-B / MagicLens-L",
], kind="train", fs=9, title_fs=11, title_color=C["train"][1])
Cb = box(ax, 106, by, bw, bh, "C. SUY LUẬN / TRUY XUẤT", [
    "Offline: mã hóa toàn bộ kho ảnh với chỉ dẫn rỗng, E(ảnh, \"\")",
    "Online: mã hóa truy vấn E(ảnh truy vấn, chỉ dẫn)",
    "Điểm = tích vô hướng giữa hai vector đã chuẩn hóa L2",
    "Sắp xếp lấy top-K ảnh",
    "Đánh giá: Recall@K (FIQ, DTIN), mAP@K (CIRCO)",
    "→ Kết quả: danh sách ảnh xếp hạng",
], kind="infer", fs=9, title_fs=11, title_color=C["infer"][1])
arrow(ax, A["r"], B["l"], "36.7M bộ ba", label_dy=1.2)
arrow(ax, B["r"], Cb["l"], "checkpoint .pkl", label_dy=1.2)
badge(ax, 2 + bw / 2, by - 3.5, "Không công bố (dùng PaLI, PaLM 2, API nội bộ)", "warn")
badge(ax, 54 + bw / 2, by - 3.5, "Repo chưa hỗ trợ huấn luyện", "warn")
badge(ax, 106 + bw / 2, by - 3.5, "Có trong repo: model.py, inference.py (chỉ bản CLIP)", "infer")
save(fig, "01_tong_quan.png")

# =====================================================================
# 2. Pipeline xây dựng dữ liệu
# =====================================================================
fig, ax = canvas(15, 7.0)
title(ax, 2, 68, "Giai đoạn A: xây dựng dữ liệu tự giám sát từ trang web",
      "Ngưỡng lấy từ Mục 3.1 và Phụ lục A của bài báo; số bộ ba cuối cùng: 36 714 118")
w, h = 44, 19
r1y, r2y = 39, 13
s1 = box(ax, 2, r1y, w, h, "1. Gom nhóm", [
    "Thu thập ảnh từ Common Crawl",
    "Ảnh có cùng URL trang được gom thành một nhóm",
    "Mỗi nhóm là một tập ứng viên để ghép cặp",
], kind="data", title_color=C["data"][1])
s2 = box(ax, 53, r1y, w, h, "2. Làm sạch", [
    "Trùng lặp: độ tương đồng ảnh CLIP > 0.98 → loại",
    "Độ phân giải < 288 × 288 → loại",
    "Ảnh quảng cáo → loại",
    "Hai nhóm chung ≥ 80% ảnh → bỏ ngẫu nhiên một nhóm",
], kind="data", title_color=C["data"][1])
s3 = box(ax, 104, r1y, w, h, "3. Mở rộng siêu dữ liệu", [
    "Alt-text: lọc bằng Google Natural Language API (token hiếm, cảm xúc, khiêu dâm)",
    "Nhãn ICA: Google Vision API, trung bình 25.2 nhãn/ảnh",
    "Chú thích ảnh: do LMM PaLI sinh",
], kind="data", title_color=C["data"][1])
s4 = box(ax, 2, r2y, w, h, "4. Chấm điểm và lọc cặp", [
    "Ghép cặp các ảnh trong cùng nhóm",
    "Liên quan thị giác: CLIP ảnh–ảnh, ngưỡng 0.82",
    "Liên quan phi thị giác: văn bản–văn bản trên chú thích, ngưỡng 0.9",
    "Ảnh đích có nhãn ICA riêng: ≥ 0.32 với ảnh đích, ≤ 0.18 với ảnh truy vấn",
    "Tối đa 3 cặp mỗi nhóm",
], kind="data", title_color=C["data"][1])
s5 = box(ax, 53, r2y, w, h, "5. Sinh chỉ dẫn bằng PaLM 2", [
    "Đầu vào: tiêu đề trang + siêu dữ liệu của hai ảnh",
    "Prompt + 5 ví dụ mẫu + bước suy luận \"Think:\"",
    "Chỉ dẫn = 1 điểm giống chung + mọi điểm khác của ảnh đích",
    "Không đủ để tìm ảnh đích nếu thiếu ảnh truy vấn",
], kind="train", title_color=C["train"][1])
s6 = box(ax, 104, r2y, w, h, "Kết quả", [
    "36 714 118 bộ ba",
    "(ảnh truy vấn, chỉ dẫn, ảnh đích)",
    "Chỉ dẫn phi mẫu, gồm cả quan hệ thị giác và ngoài thị giác",
], kind="infer", title_color=C["infer"][1])
arrow(ax, s1["r"], s2["l"])
arrow(ax, s2["r"], s3["l"])
poly_arrow(ax, [s3["b"], (126, 35), (24, 35), s4["t"]])
arrow(ax, s4["r"], s5["l"])
arrow(ax, s5["r"], s6["l"])
# Ví dụ
ex_y = 1.5
ax.text(2, ex_y + 8.2, "Ví dụ (Hình 3 của bài báo):", fontsize=8.5, fontweight="bold", color=INK, va="top")
ax.text(2, ex_y + 5.6,
        "Ảnh truy vấn: [bricklink rancor] [meta; playset; …] [a lego set with a monster and two minifigures]     "
        "Ảnh đích: [lego rancor] [tooth; brown; roar; …] [lego monster with a chain around its hand]",
        fontsize=7.8, color=INK2, va="top")
ax.text(2, ex_y + 2.6,
        "→ Chỉ dẫn sinh ra: \"Monster like this with a chain around its hand and without minifigures.\"",
        fontsize=8.2, color=C["train"][1], va="top", fontweight="bold")
save(fig, "02_du_lieu.png")

# =====================================================================
# 3. Kiến trúc mô hình (theo mã nguồn)
# =====================================================================
fig, ax = canvas(15, 10.2)
title(ax, 2, 100, "Kiến trúc mô hình MagicLens (bản CLIP, theo model.py và layers.py)",
      "B = batch, D = 512 (Base) / 768 (Large); số head: 8 (Base) / 16 (Large)")
# Hàng trên: luồng chính
y0, hh = 62, 22
img = box(ax, 2, y0 + 12, 15, 10, "Ảnh", ["[B, H, W, 3]"], kind="white", fs=8, bullet=False, align="center")
txt = box(ax, 2, y0 - 6, 15, 10, "Văn bản", ["token ids [B, 77]", "ảnh đích: \"\""], kind="white", fs=8,
          bullet=False, align="center")
pre = box(ax, 21, y0 + 9, 22, 16, "Tiền xử lý", ["cắt vùng vuông giữa", "resize 224, bilinear", "chuẩn hóa CLIP"],
          kind="neutral", fs=7.8)
vit = box(ax, 47, y0 + 9, 22, 16, "Bộ mã hóa ảnh", ["CLIP ViT-B/16 hoặc L/14", "197 token, 12/24 lớp", "→ img_embed [B, D]"],
          kind="data", fs=7.8, title_color=C["data"][1])
tenc = box(ax, 47, y0 - 9, 22, 16, "Bộ mã hóa văn bản", ["CLIP Transformer", "→ txt_embed [B, D]"],
           kind="train", fs=7.8, title_color=C["train"][1])
cat = box(ax, 74, y0 + 1, 15, 15, "Ghép chuỗi", ["[v ; t]", "[B, 2, D]"], kind="white", fs=8.5, bullet=False,
          align="center")
stk = box(ax, 93, y0 + 1, 18, 15, "StackedTransformer", ["4 khối", "self-attention"], kind="model", fs=8.5,
          bullet=False, align="center", title_color=C["model"][1])
pool = box(ax, 115, y0 + 1, 17, 15, "Attention pooling", ["1 query học được", "→ [B, D]"], kind="model", fs=8.5,
           bullet=False, align="center", title_color=C["model"][1])
l2 = box(ax, 136, y0 + 1, 12, 15, "Chuẩn hóa L2", ["r = r / ‖r‖"], kind="white", fs=8.5, bullet=False,
         align="center")
arrow(ax, img["r"], pre["l"])
arrow(ax, pre["r"], vit["l"])
poly_arrow(ax, [txt["r"], (45, txt["r"][1]), (45, tenc["l"][1]), tenc["l"]])
arrow(ax, vit["r"], (cat["l"][0], cat["l"][1] + 3))
arrow(ax, tenc["r"], (cat["l"][0], cat["l"][1] - 3))
arrow(ax, cat["r"], stk["l"])
arrow(ax, stk["r"], pool["l"])
arrow(ax, pool["r"], l2["l"])
ax.text(142, y0 - 1.5, "multimodal_embed_norm", ha="center", va="top", fontsize=7.5, color=MUTED)
ax.text(58, y0 + 27.5, "Backbone: 149.6M / 427.6M tham số (học tiếp, lr 2e-6)", ha="center", fontsize=8,
        color=MUTED)
ax.text(112.5, y0 + 18.5, "Phần mới: 16.8M / 37.8M tham số (lr 2e-5)", ha="center", fontsize=8, color=MUTED)
ax.text(32, y0 + 7.5, "model.py:99", ha="center", fontsize=7, color=MUTED)
ax.text(58, y0 - 10.5, "model.py:112 (clip_encode, normalize=False)", ha="center", fontsize=7, color=MUTED)
ax.text(101.5, y0 - 0.5, "model.py:175–183", ha="center", va="top", fontsize=7, color=MUTED)

# Hàng dưới: phóng to 1 khối + pooling
ax.add_patch(Rectangle((2, 3), 146, 42, fc="none", ec=LINE, lw=1, ls=(0, (4, 3))))
ax.text(4, 43.5, "Chi tiết một khối self-attention (layers.py: Transformer) và lớp pooling (AttenTokenPoolingLayer)",
        fontsize=9, fontweight="bold", color=INK, va="top")
yb = 18
k_in = box(ax, 4, yb, 12, 12, "x", ["[B, 2, D]"], kind="white", fs=8, bullet=False, align="center")
k_ln1 = box(ax, 20, yb, 13, 12, "LayerNorm", ["x̂ · (1 + γ) + β"], kind="neutral", fs=7.5, bullet=False,
            align="center")
k_att = box(ax, 37, yb - 2, 24, 16, "Multi-head attention", ["Q, K, V chiếu từ x̂", "logit = 50·tanh(logit/50)",
                                                              "softmax trên 2 vị trí"],
            kind="model", fs=7.5, title_color=C["model"][1])
k_add1 = box(ax, 65, yb + 2, 8, 8, "+", [], kind="white", fs=10, va_center=True)
k_ln2 = box(ax, 77, yb, 13, 12, "LayerNorm", ["(1 + γ)"], kind="neutral", fs=7.5, bullet=False, align="center")
k_ffn = box(ax, 94, yb, 17, 12, "FFN", ["D → 4D → D", "ReLU"], kind="model", fs=7.5, bullet=False, align="center",
            title_color=C["model"][1])
k_add2 = box(ax, 115, yb + 2, 8, 8, "+", [], kind="white", fs=10, va_center=True)
k_out = box(ax, 127, yb, 19, 12, "Lặp 4 lần", ["rồi vào pooling"], kind="white", fs=7.5, bullet=False,
            align="center")
for a, b in [(k_in, k_ln1), (k_ln1, k_att), (k_att, k_add1), (k_add1, k_ln2), (k_ln2, k_ffn), (k_ffn, k_add2),
             (k_add2, k_out)]:
    arrow(ax, a["r"], b["l"], lw=1.1)
poly_arrow(ax, [(10, yb + 12), (10, yb + 17), (69, yb + 17), k_add1["t"]], color=MUTED, lw=1)
poly_arrow(ax, [(71, yb + 10), (71, yb + 15.5), (119, yb + 15.5), k_add2["t"]], color=MUTED, lw=1)
ax.text(40, yb + 17.6, "kết nối tắt (residual)", ha="center", fontsize=7, color=MUTED)
ax.text(4, 11.5, "Pooling: query q học được [1, D] chú ý lên 2 token; chiều ẩn 4D, hệ số tỉ lệ PerDimScale (softplus); "
                 "sau đó LayerNorm → lấy [:, 0] → [B, D].", fontsize=8, color=INK2, va="top")
ax.text(4, 7.5, "Vì chuỗi chỉ có 2 token, mỗi head cho ra o = α·W_V·v̂ + (1 − α)·W_V·t̂ với α ∈ (0, 1): "
                "khối hợp nhất về bản chất là phép trộn có cổng giữa hai vector toàn cục.",
        fontsize=8, color=INK2, va="top")
save(fig, "03_mo_hinh.png")

# =====================================================================
# 4. Huấn luyện
# =====================================================================
fig, ax = canvas(15, 7.4)
title(ax, 2, 72, "Giai đoạn B: huấn luyện tương phản với mẫu âm là ảnh truy vấn",
      "Minh họa với N = 4; thực tế N = 2048 nên mỗi truy vấn có 4095 mẫu âm")
q = box(ax, 2, 50, 30, 14, "Truy vấn (N cặp)", ["(ảnh truy vấn j, chỉ dẫn j)"], kind="data", fs=8.5, bullet=False,
        align="center", title_color=C["data"][1])
t = box(ax, 2, 31, 30, 14, "Ảnh đích (N cặp)", ["(ảnh đích j, \"\")"], kind="infer", fs=8.5, bullet=False,
        align="center", title_color=C["infer"][1])
n = box(ax, 2, 12, 30, 14, "Mẫu âm truy vấn (N cặp)", ["(ảnh truy vấn j, \"\")"], kind="train", fs=8.5,
        bullet=False, align="center", title_color=C["train"][1])
E = box(ax, 40, 12, 16, 52, "Mô hình E", ["dùng chung", "tham số"], kind="model", fs=8.5, bullet=False,
        align="center", title_color=C["model"][1], va_center=True)
for b, lab in [(q, "r_q"), (t, "r_t"), (n, "r_t′")]:
    arrow(ax, b["r"], (E["l"][0], b["r"][1]))
    arrow(ax, (E["r"][0], b["r"][1]), (63, b["r"][1]), lab, label_dy=0.6)
# Ma trận tương đồng
cs, gx, gy = 4.2, 72, 22
ax.text(gx + 4 * cs / 2, gy + 4 * cs + 9.5, "sim(r_q, r_t) / τ", ha="center", fontsize=9, color=INK)
ax.text(gx + 4 * cs + 3 + 4 * cs / 2, gy + 4 * cs + 9.5, "sim(r_q, r_t′) / τ", ha="center", fontsize=9, color=INK)
for i in range(4):
    ax.text(gx - 2, gy + (3 - i) * cs + cs / 2, f"q{i + 1}", ha="right", va="center", fontsize=8, color=INK2)
    for j in range(4):
        pos = i == j
        ax.add_patch(Rectangle((gx + j * cs, gy + (3 - i) * cs), cs, cs, fc="#1baf7a" if pos else "#f5f4f0",
                               ec="white", lw=1.5))
        hard = i == j
        ax.add_patch(Rectangle((gx + 4 * cs + 3 + j * cs, gy + (3 - i) * cs), cs, cs,
                               fc="#eb6834" if hard else "#f8d3c1", ec="white", lw=1.5))
    ax.text(gx + i * cs + cs / 2, gy + 4 * cs + 1.4, f"t{i + 1}", ha="center", fontsize=8, color=INK2)
    ax.text(gx + 4 * cs + 3 + i * cs + cs / 2, gy + 4 * cs + 1.4, f"t′{i + 1}", ha="center", fontsize=8, color=INK2)
ax.text(gx + 4 * cs + 1.5, gy - 3, "softmax theo từng hàng (2N số hạng)", ha="center", va="top", fontsize=8.5,
        color=INK2)
# Chú giải
lx, ly = 112, 56
for k, (fc, txt_) in enumerate([("#1baf7a", "Cặp dương: (q_i, t_i)"),
                                ("#f5f4f0", "Mẫu âm: ảnh đích của truy vấn khác"),
                                ("#eb6834", "Mẫu âm khó: chính ảnh truy vấn của q_i"),
                                ("#f8d3c1", "Mẫu âm: ảnh truy vấn của truy vấn khác")]):
    ax.add_patch(Rectangle((lx, ly - k * 4.2), 3, 3, fc=fc, ec=LINE, lw=0.6))
    ax.text(lx + 4.5, ly - k * 4.2 + 1.5, txt_, va="center", fontsize=8.5, color=INK)
box(ax, 110, 12, 38, 19, "Thiết lập huấn luyện", [
    "Batch N = 2048, τ học được (khởi tạo 0.07)",
    "Adafactor; lr 2e-5 (phần mới), 2e-6 (backbone)",
    "50 000 bước ≈ 2.8 epoch trên 36.7M bộ ba",
    "64 TPU (Base) / 128 TPU (Large), ~6 giờ",
    "Chọn checkpoint theo val CIRR + CIRCO",
], kind="neutral", fs=8)
ax.text(2, 6.5, "ℒ_i = −log [ exp(s_ii) / Σ_j ( exp(s_ij) + exp(s′_ij) ) ]", fontsize=11, color=INK,
        va="center", family="DejaVu Sans")
ax.text(2, 2.5, "Bỏ mẫu âm truy vấn: CIRR R@1 giảm 31.6 → 1.6 vì mô hình học lối tắt \"trả về chính ảnh truy vấn\".",
        fontsize=8.2, color=C["train"][1], va="center")
save(fig, "04_huan_luyen.png")

# =====================================================================
# 5. Suy luận / đánh giá trong repo
# =====================================================================
fig, ax = canvas(15, 6.6)
title(ax, 2, 64, "Giai đoạn C: truy xuất và đánh giá (inference.py, data_utils.py)",
      "Phần offline chạy một lần cho cả kho ảnh; phần online chạy cho từng lô truy vấn")
ax.add_patch(Rectangle((1, 36), 148, 20, fc="none", ec=C["infer"][1], lw=1, ls=(0, (4, 3))))
ax.text(3, 54.6, "OFFLINE: xây chỉ mục", fontsize=9, fontweight="bold", color=C["infer"][1], va="top")
ax.add_patch(Rectangle((1, 2), 148, 31, fc="none", ec=C["data"][1], lw=1, ls=(0, (4, 3))))
ax.text(3, 31.6, "ONLINE: xử lý truy vấn", fontsize=9, fontweight="bold", color=C["data"][1], va="top")
k1 = box(ax, 4, 39.5, 24, 11, "Kho ảnh (M ảnh)", ["FIQ ~5K, DTIN ~17K,", "CIRCO 123 403"], kind="white", fs=8,
         bullet=False, align="center")
k2 = box(ax, 34, 39.5, 28, 11, "process_img", ["resize thẳng 224 × 224", "chia cho giá trị max của ảnh"],
         kind="warn", fs=8, title_color="#a06c00")
k3 = box(ax, 68, 39.5, 26, 11, "E(ảnh, \"\")", ["theo lô 1024", "token của chuỗi rỗng"], kind="model", fs=8,
         title_color=C["model"][1])
k4 = box(ax, 100, 39.5, 30, 11, "index_embeddings", ["[M, D], đã chuẩn hóa L2", "giữ trong bộ nhớ"], kind="infer",
         fs=8, title_color=C["infer"][1])
for a, b in [(k1, k2), (k2, k3), (k3, k4)]:
    arrow(ax, a["r"], b["l"])
q1 = box(ax, 4, 9, 24, 12, "Truy vấn", ["ảnh tham chiếu", "+ văn bản của benchmark"], kind="white", fs=8,
         bullet=False, align="center")
q2 = box(ax, 34, 6, 28, 18, "Mẫu chỉ dẫn", ["FIQ: \"c1 and c2\"", "CIRCO: \"find {shared_concept} but "
                                            "{relative_caption}\"", "DTIN: \"find this object in {domain}\""],
         kind="neutral", fs=7.6)
q3 = box(ax, 68, 9, 26, 12, "E(ảnh, chỉ dẫn)", ["→ r_q [B, D]"], kind="model", fs=8, bullet=False,
         align="center", title_color=C["model"][1])
q4 = box(ax, 100, 9, 22, 12, "Điểm số", ["r_q · indexᵀ  [B, M]", "argsort → top-50"], kind="data", fs=8,
         bullet=False, align="center", title_color=C["data"][1])
for a, b in [(q1, q2), (q2, q3), (q3, q4)]:
    arrow(ax, a["r"], b["l"])
arrow(ax, k4["b"], (k4["b"][0] - 4, q4["t"][1]), "so khớp", label_dy=0.5)
ev = box(ax, 126, 5, 21, 21, "Đánh giá", ["FIQ, DTIN: Recall@10, @50 tại máy",
                                         "CIRCO: ghi top-50 ra JSON, nộp máy chủ chấm mAP@K"],
         kind="neutral", fs=7.8)
arrow(ax, q4["r"], (ev["l"][0], q4["r"][1]))
ax.text(48, 38.6, "Lưu ý: bóp méo tỉ lệ khung hình; CIRCO cần ~74 GB RAM nếu giữ toàn bộ ảnh", ha="center",
        va="top", fontsize=7.4, color="#a06c00")
save(fig, "05_suy_luan.png")
