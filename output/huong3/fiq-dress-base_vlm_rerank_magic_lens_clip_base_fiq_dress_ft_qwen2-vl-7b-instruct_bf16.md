# Rerank top-K của MagicLens-base bằng VLM trên FashionIQ dress (bước 1: MagicLens tinh chỉnh), VLM `Qwen/Qwen2-VL-7B-Instruct` (bfloat16)

VLM: `Qwen/Qwen2-VL-7B-Instruct` (bfloat16), chấm 50 ứng viên đầu của mỗi truy vấn bằng log-odds logit("Yes") − logit("No"). Điểm mới = z(MagicLens) + w · z(VLM) trên k ứng viên đầu. Truy vấn: dev 930 / test 931 (cùng cách chia với fiq_experiments.py). Chọn k và w theo R_avg của dev. Bước 1 là MagicLens đã tinh chỉnh trên tập train của FashionIQ (`magic_lens_clip_base_fiq_dress_ft.pkl`); VLM không được huấn luyện thêm, và nhãn của nửa dev chỉ dùng để chọn k và w.

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| MagicLens tinh chỉnh (bước 1) | 45.3 | 11.4 | 25.7 | 34.2 | 57.9 | 46.0 |
| Chỉ VLM, rerank top-50 | 48.4 | 14.7 | 30.3 | 38.8 | 57.9 | 48.3 |
| MagicLens + VLM, top-50 (w=2) | 49.8 | 19.1 | 34.2 | 41.8 | 57.9 | 49.8 |

Chênh lệch so với MagicLens tinh chỉnh trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%): R@10 +7.6 [+5.5, +9.8], R_avg +3.8 [+2.7, +4.9] điểm. Rerank trong top-k (k ≤ 50) không đổi tập top-50, nên R@50 giữ nguyên, R_avg chỉ đổi theo R@10 và khoảng tin cậy của R_avg đúng bằng một nửa của R@10. Trên test, ảnh đích được đưa vào top-10 ở 94 truy vấn và bị đẩy ra ở 23 truy vấn.

## Theo độ sâu rerank k (w tốt nhất trên dev cho từng k)

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| top-20 (w=1.5); trần R@10 test 43.7 | 47.8 | 18.3 | 32.7 | 39.0 | 57.9 | 48.4 |
| top-30 (w=2); trần R@10 test 50.7 | 49.2 | 18.8 | 33.4 | 40.9 | 57.9 | 49.4 |
| top-50 (w=2); trần R@10 test 57.9 | 49.8 | 19.1 | 34.2 | 41.8 | 57.9 | 49.8 |
| top-20, chỉ VLM | 47.3 | 16.1 | 31.8 | 37.4 | 57.9 | 47.6 |
| top-30, chỉ VLM | 48.1 | 15.5 | 31.5 | 39.4 | 57.9 | 48.7 |
| top-50, chỉ VLM | 48.4 | 14.7 | 30.3 | 38.8 | 57.9 | 48.3 |

## So sánh thêm trên test (R@10, bootstrap như trên)

- Chỉ VLM (top-50) so với MagicLens tinh chỉnh: +4.6 [+1.9, +7.5].
- MagicLens + VLM so với chỉ VLM: +3.0 [+1.2, +4.8].
- Kho ảnh chứa cả ảnh truy vấn (giống cách đánh giá gốc). Ảnh này nằm trong top-10 của 121 truy vấn test trước rerank, còn 34 sau rerank. Nếu lọc ảnh truy vấn khỏi danh sách kết quả của cả hai hệ: R@10 34.26 → 41.78, chênh lệch +7.5 [+5.4, +9.7].

## Phân tích trên test

- Khả năng tách ảnh đích khỏi 49 ứng viên còn lại, trên 539 truy vấn có ảnh đích trong top-50 (AUC trung bình theo truy vấn; 0.5 là ngẫu nhiên): MagicLens 0.761, VLM 0.813, MagicLens + VLM 0.838.
- Tương quan Pearson giữa z(MagicLens) và z(VLM) trên mọi cặp (truy vấn, ứng viên) trong top-50 của 931 truy vấn test: 0.21.

## Toàn bộ lưới: R@10 dev / test (chỉ để tham khảo; k và w chỉ được chọn trên dev)

| w | top-20 | top-30 | top-50 |
|---|---|---|---|
| MagicLens tinh chỉnh | 31.3 / 34.2 | 31.3 / 34.2 | 31.3 / 34.2 |
| w=0.1 | 32.0 / 34.3 | 32.3 / 34.3 | 32.3 / 34.3 |
| w=0.2 | 32.6 / 34.9 | 32.7 / 35.0 | 32.9 / 35.0 |
| w=0.3 | 33.3 / 35.8 | 33.3 / 36.1 | 34.1 / 36.6 |
| w=0.5 | 35.6 / 37.3 | 35.9 / 37.9 | 36.8 / 38.5 |
| w=0.7 | 35.8 / 38.3 | 37.4 / 39.5 | 38.5 / 40.3 |
| w=1 | 36.2 / 38.9 | 38.8 / 40.3 | 39.6 / 41.2 |
| w=1.5 | 36.5 / 39.0 | 39.0 / 41.1 | 40.0 / 42.3 |
| w=2 | 36.5 / 39.1 | 39.2 / 40.9 | 40.3 / 41.8 |
| w=3 | 36.3 / 39.3 | 38.4 / 40.1 | 40.0 / 40.8 |
| w=5 | 36.1 / 38.8 | 38.1 / 39.7 | 40.0 / 41.1 |
| chỉ VLM | 35.3 / 37.4 | 37.0 / 39.4 | 37.6 / 38.8 |

## Chi phí

Thời gian chấm VLM: 0.9 giờ GPU cho 93050 cặp (34 ms/cặp, gồm cả đọc ảnh, không gồm tải và nạp mô hình).
