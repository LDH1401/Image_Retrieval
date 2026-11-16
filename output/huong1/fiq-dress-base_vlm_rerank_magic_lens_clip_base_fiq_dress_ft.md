# Rerank top-K của MagicLens-base bằng VLM trên FashionIQ dress (bước 1: MagicLens tinh chỉnh)

VLM: `Qwen/Qwen2-VL-2B-Instruct` (float16), chấm 50 ứng viên đầu của mỗi truy vấn bằng log-odds logit("Yes") − logit("No"). Điểm mới = z(MagicLens) + w · z(VLM) trên k ứng viên đầu. Truy vấn: dev 930 / test 931 (cùng cách chia với fiq_experiments.py). Chọn k và w theo R_avg của dev. Bước 1 là MagicLens đã tinh chỉnh trên tập train của FashionIQ (`magic_lens_clip_base_fiq_dress_ft.pkl`); VLM không được huấn luyện thêm, và nhãn của nửa dev chỉ dùng để chọn k và w.

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| MagicLens tinh chỉnh (bước 1) | 45.3 | 11.4 | 25.7 | 34.2 | 57.9 | 46.0 |
| Chỉ VLM, rerank top-20 | 46.5 | 10.7 | 25.3 | 35.2 | 57.9 | 46.6 |
| MagicLens + VLM, top-50 (w=1.5) | 48.2 | 15.1 | 29.8 | 36.6 | 57.9 | 47.3 |

Chênh lệch so với MagicLens tinh chỉnh trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%): R@10 +2.5 [+0.5, +4.4], R_avg +1.2 [+0.3, +2.2] điểm. Rerank trong top-k (k ≤ 50) không đổi tập top-50, nên R@50 giữ nguyên, R_avg chỉ đổi theo R@10 và khoảng tin cậy của R_avg đúng bằng một nửa của R@10. Trên test, ảnh đích được đưa vào top-10 ở 59 truy vấn và bị đẩy ra ở 36 truy vấn.

## Theo độ sâu rerank k (w tốt nhất trên dev cho từng k)

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| top-20 (w=1.5); trần R@10 test 43.7 | 47.5 | 14.6 | 29.3 | 36.4 | 57.9 | 47.2 |
| top-30 (w=1.5); trần R@10 test 50.7 | 48.0 | 14.7 | 30.0 | 36.4 | 57.9 | 47.2 |
| top-50 (w=1.5); trần R@10 test 57.9 | 48.2 | 15.1 | 29.8 | 36.6 | 57.9 | 47.3 |
| top-20, chỉ VLM | 46.5 | 10.7 | 25.3 | 35.2 | 57.9 | 46.6 |
| top-30, chỉ VLM | 46.5 | 9.1 | 23.4 | 34.3 | 57.9 | 46.1 |
| top-50, chỉ VLM | 45.3 | 8.7 | 21.7 | 31.6 | 57.9 | 44.7 |

## So sánh thêm trên test (R@10, bootstrap như trên)

- Chỉ VLM (top-20) so với MagicLens tinh chỉnh: +1.1 [-1.1, +3.1].
- MagicLens + VLM so với chỉ VLM: +1.4 [-0.3, +3.1].
- Kho ảnh chứa cả ảnh truy vấn (giống cách đánh giá gốc). Ảnh này nằm trong top-10 của 121 truy vấn test trước rerank, còn 33 sau rerank. Nếu lọc ảnh truy vấn khỏi danh sách kết quả của cả hai hệ: R@10 34.26 → 36.63, chênh lệch +2.4 [+0.4, +4.3].

## Phân tích trên test

- Khả năng tách ảnh đích khỏi 49 ứng viên còn lại, trên 539 truy vấn có ảnh đích trong top-50 (AUC trung bình theo truy vấn; 0.5 là ngẫu nhiên): MagicLens 0.761, VLM 0.736, MagicLens + VLM 0.796.
- Tương quan Pearson giữa z(MagicLens) và z(VLM) trên mọi cặp (truy vấn, ứng viên) trong top-50 của 931 truy vấn test: 0.17.

## Toàn bộ lưới: R@10 dev / test (chỉ để tham khảo; k và w chỉ được chọn trên dev)

| w | top-20 | top-30 | top-50 |
|---|---|---|---|
| MagicLens tinh chỉnh | 31.3 / 34.2 | 31.3 / 34.2 | 31.3 / 34.2 |
| w=0.1 | 31.6 / 33.8 | 31.4 / 33.8 | 31.5 / 33.8 |
| w=0.2 | 31.8 / 34.3 | 31.8 / 34.5 | 32.5 / 34.7 |
| w=0.3 | 32.7 / 34.6 | 32.8 / 35.0 | 33.1 / 35.2 |
| w=0.5 | 33.7 / 35.9 | 34.3 / 36.0 | 34.9 / 35.9 |
| w=0.7 | 35.1 / 35.8 | 35.2 / 36.1 | 35.7 / 36.0 |
| w=1 | 35.4 / 36.2 | 35.6 / 36.3 | 36.1 / 36.6 |
| w=1.5 | 35.8 / 36.4 | 36.7 / 36.4 | 37.2 / 36.6 |
| w=2 | 35.1 / 36.4 | 36.1 / 36.4 | 36.5 / 36.9 |
| w=3 | 35.3 / 36.1 | 35.4 / 36.7 | 36.0 / 36.5 |
| w=5 | 34.5 / 36.0 | 35.3 / 36.3 | 34.7 / 35.6 |
| chỉ VLM | 33.8 / 35.2 | 33.7 / 34.3 | 31.4 / 31.6 |

## Chi phí

Thời gian chấm VLM: 0.9 giờ GPU cho 93050 cặp (36 ms/cặp, gồm cả đọc ảnh, không gồm tải và nạp mô hình).
