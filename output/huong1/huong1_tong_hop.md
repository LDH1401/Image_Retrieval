# Hướng 1: ghép tinh chỉnh với rerank bằng VLM (FashionIQ dress, MagicLens-base)

Bước 1 tinh chỉnh: `magic_lens_clip_base_fiq_dress_ft.pkl`. VLM: `Qwen/Qwen2-VL-2B-Instruct` (float16), chấm top-50 của mỗi truy vấn, cùng cấu hình với lần rerank zero-shot. Kho ảnh 3653 ảnh; truy vấn dev 930 / test 931. k và w của (b), (d), (e) được vlm_rerank_eval.py chọn trên dev; mọi con số dưới đây là trên test, trừ cột dev R_avg.

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| (a) MagicLens zero-shot | 32.5 | 8.1 | 17.8 | 23.3 | 42.1 | 32.7 |
| (b) (a) + rerank VLM, top-50 (w=3) | 35.5 | 10.8 | 21.7 | 28.5 | 42.1 | 35.3 |
| (c) MagicLens tinh chỉnh | 45.3 | 11.4 | 25.7 | 34.2 | 57.9 | 46.0 |
| (d) (c), xếp lại chỉ theo VLM, top-20 | 46.5 | 10.7 | 25.3 | 35.2 | 57.9 | 46.6 |
| (e) (c) + rerank VLM, top-50 (w=1.5) | 48.2 | 15.1 | 29.8 | 36.6 | 57.9 | 47.3 |

## So sánh trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%, đơn vị điểm %)

| So sánh | ΔR@10 | ΔR_avg |
|---|---|---|
| (e) − (c): rerank khi bước 1 đã tinh chỉnh | +2.5 [+0.5, +4.4] | +1.2 [+0.3, +2.2] |
| (b) − (a): rerank khi bước 1 là zero-shot | +5.2 [+3.0, +7.2] | +2.6 [+1.5, +3.6] |
| [(e) − (c)] − [(b) − (a)]: mức tăng nhờ rerank thay đổi bao nhiêu | -2.7 [-5.3, -0.2] | -1.3 [-2.6, -0.1] |
| (d) − (c): chỉ VLM so với bước 1 đã tinh chỉnh | +1.1 [-1.1, +3.1] | +0.5 [-0.5, +1.6] |
| (e) − (b): thêm tinh chỉnh khi đã có rerank | +8.2 [+5.8, +10.4] | +12.0 [+9.8, +14.1] |
| (e) − (a): cả hai cải tiến so với zero-shot | +13.3 [+10.7, +16.0] | +14.6 [+12.4, +16.7] |

Rerank chỉ xếp lại top-50 nên không đổi R@50: trần R@10 sau rerank là R@50 của bước 1 (42.1 với zero-shot, 57.9 với bản tinh chỉnh).

## Hai điểm số còn bổ sung cho nhau không (test)

AUC: khả năng tách ảnh đích khỏi các ứng viên khác trong top-k, trung bình trên các truy vấn có ảnh đích trong top-k (0.5 là ngẫu nhiên). Tương quan: Pearson giữa z(MagicLens) và z(VLM) trên mọi cặp (truy vấn, ứng viên) trong top-k.

| Bước 1 | k | Truy vấn có ảnh đích trong top-k | AUC MagicLens | AUC VLM | AUC trộn | Tương quan |
|---|---|---|---|---|---|---|
| Zero-shot | 50 | 392 | 0.739 | 0.781 | 0.812 | 0.15 |
| Tinh chỉnh | 50 | 539 | 0.761 | 0.736 | 0.796 | 0.17 |

Ảnh đích được rerank đưa vào top-10 / đẩy ra khỏi top-10: zero-shot 77 / 29, tinh chỉnh 59 / 36 truy vấn.
