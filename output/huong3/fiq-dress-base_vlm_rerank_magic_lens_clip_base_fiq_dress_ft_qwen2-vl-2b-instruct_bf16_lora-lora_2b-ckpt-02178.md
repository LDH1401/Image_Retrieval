# Rerank top-K của MagicLens-base bằng VLM trên FashionIQ dress (bước 1: MagicLens tinh chỉnh), VLM `Qwen/Qwen2-VL-2B-Instruct` (bfloat16) + LoRA `lora_2b/ckpt-02178`

VLM: `Qwen/Qwen2-VL-2B-Instruct` (bfloat16) + LoRA `lora_2b/ckpt-02178`, chấm 50 ứng viên đầu của mỗi truy vấn bằng log-odds logit("Yes") − logit("No"). Điểm mới = z(MagicLens) + w · z(VLM) trên k ứng viên đầu. Truy vấn: dev 930 / test 931 (cùng cách chia với fiq_experiments.py). Chọn k và w theo R_avg của dev. Bước 1 là MagicLens đã tinh chỉnh trên tập train của FashionIQ (`magic_lens_clip_base_fiq_dress_ft.pkl`); VLM đã được tinh chỉnh LoRA trên tập train đó (`lora_2b/ckpt-02178`). Nhãn của nửa dev dùng để chọn checkpoint LoRA (vlm_lora_train.py), k và w.

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| MagicLens tinh chỉnh (bước 1) | 45.3 | 11.4 | 25.7 | 34.2 | 57.9 | 46.0 |
| Chỉ VLM, rerank top-50 | 49.9 | 15.9 | 32.8 | 40.7 | 57.9 | 49.3 |
| MagicLens + VLM, top-50 (w=2) | 51.4 | 18.9 | 37.2 | 45.9 | 57.9 | 51.9 |

Chênh lệch so với MagicLens tinh chỉnh trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%): R@10 +11.7 [+9.5, +14.2], R_avg +5.9 [+4.7, +7.1] điểm. Rerank trong top-k (k ≤ 50) không đổi tập top-50, nên R@50 giữ nguyên, R_avg chỉ đổi theo R@10 và khoảng tin cậy của R_avg đúng bằng một nửa của R@10. Trên test, ảnh đích được đưa vào top-10 ở 124 truy vấn và bị đẩy ra ở 15 truy vấn.

## Theo độ sâu rerank k (w tốt nhất trên dev cho từng k)

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| top-20 (w=5); trần R@10 test 43.7 | 48.8 | 19.2 | 32.9 | 40.2 | 57.9 | 49.0 |
| top-30 (w=1.5); trần R@10 test 50.7 | 50.1 | 18.9 | 35.8 | 43.8 | 57.9 | 50.9 |
| top-50 (w=2); trần R@10 test 57.9 | 51.4 | 18.9 | 37.2 | 45.9 | 57.9 | 51.9 |
| top-20, chỉ VLM | 48.2 | 16.2 | 31.9 | 39.2 | 57.9 | 48.5 |
| top-30, chỉ VLM | 49.4 | 17.2 | 33.3 | 41.4 | 57.9 | 49.6 |
| top-50, chỉ VLM | 49.9 | 15.9 | 32.8 | 40.7 | 57.9 | 49.3 |

## So sánh thêm trên test (R@10, bootstrap như trên)

- Chỉ VLM (top-50) so với MagicLens tinh chỉnh: +6.6 [+3.7, +9.7].
- MagicLens + VLM so với chỉ VLM: +5.2 [+3.1, +7.2].
- Kho ảnh chứa cả ảnh truy vấn (giống cách đánh giá gốc). Ảnh này nằm trong top-10 của 121 truy vấn test trước rerank, còn 1 sau rerank. Nếu lọc ảnh truy vấn khỏi danh sách kết quả của cả hai hệ: R@10 34.26 → 45.86, chênh lệch +11.6 [+9.3, +14.1].

## Phân tích trên test

- Khả năng tách ảnh đích khỏi 49 ứng viên còn lại, trên 539 truy vấn có ảnh đích trong top-50 (AUC trung bình theo truy vấn; 0.5 là ngẫu nhiên): MagicLens 0.761, VLM 0.841, MagicLens + VLM 0.870.
- Tương quan Pearson giữa z(MagicLens) và z(VLM) trên mọi cặp (truy vấn, ứng viên) trong top-50 của 931 truy vấn test: 0.13.

## Toàn bộ lưới: R@10 dev / test (chỉ để tham khảo; k và w chỉ được chọn trên dev)

| w | top-20 | top-30 | top-50 |
|---|---|---|---|
| MagicLens tinh chỉnh | 31.3 / 34.2 | 31.3 / 34.2 | 31.3 / 34.2 |
| w=0.1 | 32.3 / 34.8 | 32.4 / 34.8 | 32.6 / 34.7 |
| w=0.2 | 33.7 / 35.7 | 33.4 / 36.0 | 34.2 / 36.3 |
| w=0.3 | 34.3 / 36.5 | 34.6 / 36.7 | 35.5 / 37.4 |
| w=0.5 | 35.5 / 37.6 | 37.0 / 39.2 | 37.8 / 39.8 |
| w=0.7 | 36.8 / 38.7 | 38.3 / 40.8 | 39.4 / 41.8 |
| w=1 | 37.2 / 39.6 | 39.9 / 42.5 | 41.8 / 44.0 |
| w=1.5 | 38.0 / 40.7 | 41.0 / 43.8 | 43.1 / 45.2 |
| w=2 | 38.2 / 40.4 | 41.0 / 43.4 | 43.5 / 45.9 |
| w=3 | 38.2 / 40.3 | 41.0 / 43.5 | 42.9 / 44.7 |
| w=5 | 38.4 / 40.2 | 40.2 / 42.5 | 42.2 / 43.5 |
| chỉ VLM | 37.1 / 39.2 | 39.6 / 41.4 | 40.5 / 40.7 |

## Chi phí

Thời gian chấm VLM: 0.3 giờ GPU cho 93050 cặp (13 ms/cặp, gồm cả đọc ảnh, không gồm tải và nạp mô hình).
