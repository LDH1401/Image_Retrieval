# Tinh chỉnh khối hợp nhất MagicLens-base trên FashionIQ dress train

Train: 5441 truy vấn, 10878 ảnh (backbone CLIP đóng băng; học 16.8M tham số). Đánh giá: dev 930 / test 931 truy vấn của tập val, như fiq_experiments.py. Chọn tốc độ học và epoch theo R_avg của dev.

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| MagicLens zero-shot (baseline) | 32.5 | 8.1 | 17.8 | 23.3 | 42.1 | 32.7 |
| Tinh chỉnh (lr=1e-05, epoch 14) | 45.3 | 11.5 | 25.7 | 34.2 | 57.9 | 46.0 |

Chênh lệch trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%): R@10 +10.8 [+8.3, +13.3], R_avg +13.3 [+11.2, +15.5] điểm.

## Đường học (dev R_avg theo epoch)

| Epoch | lr=1e-05 |
|---|---|
| 1 | 39.5 |
| 2 | 41.2 |
| 3 | 41.6 |
| 4 | 43.7 |
| 5 | 43.8 |
| 6 | 43.7 |
| 7 | 44.1 |
| 8 | 44.1 |
| 9 | 43.4 |
| 10 | 44.6 |
| 11 | 44.9 |
| 12 | 44.0 |
| 13 | 44.6 |
| 14 | 45.3 |
| 15 | 44.6 |

Thời gian huấn luyện (CPU, mọi lần thử, gồm đánh giá mỗi epoch): 16.5 phút.

Tham số của cấu hình được chọn đã lưu vào `models/magic_lens_clip_base_fiq_dress_ft.pkl`, cùng định dạng với checkpoint gốc (nạp được bằng `inference.load_model`).
