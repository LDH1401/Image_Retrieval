# Hướng 3: reranker mạnh hơn trên cùng tập ứng viên (FashionIQ dress, MagicLens-base)

Bước 1: `magic_lens_clip_base_fiq_dress_ft.pkl` (top-50, kho 3653 ảnh, dev 930 / test 931 truy vấn). Chỉ VLM thay đổi. k và w của mỗi VLM được vlm_rerank_eval.py chọn trên dev; mọi số dưới đây là trên test, trừ cột dev R_avg.

| Cấu hình | dev R_avg | test R@1 | test R@5 | test R@10 | test R@50 | test R_avg |
|---|---|---|---|---|---|---|
| (a) MagicLens zero-shot | 32.5 | 8.1 | 17.8 | 23.3 | 42.1 | 32.7 |
| (c) MagicLens tinh chỉnh (bước 1) | 45.3 | 11.4 | 25.7 | 34.2 | 57.9 | 46.0 |
| Qwen2-VL-2B fp16, chỉ VLM (top-20) | 46.5 | 10.7 | 25.3 | 35.2 | 57.9 | 46.6 |
| MagicLens + Qwen2-VL-2B fp16 (top-50, w=1.5) | 48.2 | 15.1 | 29.8 | 36.6 | 57.9 | 47.3 |
| Qwen2-VL-7B bf16, chỉ VLM (top-50) | 48.4 | 14.7 | 30.3 | 38.8 | 57.9 | 48.3 |
| MagicLens + Qwen2-VL-7B bf16 (top-50, w=2) | 49.8 | 19.1 | 34.2 | 41.8 | 57.9 | 49.8 |
| Qwen2-VL-2B bf16 + LoRA, chỉ VLM (top-50) | 49.9 | 15.9 | 32.8 | 40.7 | 57.9 | 49.3 |
| MagicLens + Qwen2-VL-2B bf16 + LoRA (top-50, w=2) | 51.4 | 18.9 | 37.2 | 45.9 | 57.9 | 51.9 |

## So sánh trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%)

| So sánh | ΔR@1 | ΔR@10 |
|---|---|---|
| MagicLens + Qwen2-VL-2B fp16 − (c) | +3.8 [+1.8, +5.7] | +2.5 [+0.5, +4.4] |
| Qwen2-VL-2B fp16 chỉ VLM − (c) | -0.6 [-3.1, +1.8] | +1.1 [-1.1, +3.1] |
| MagicLens + Qwen2-VL-7B bf16 − (c) | +7.7 [+5.5, +9.9] | +7.6 [+5.5, +9.8] |
| Qwen2-VL-7B bf16 chỉ VLM − (c) | +3.3 [+0.8, +5.9] | +4.6 [+1.9, +7.5] |
| MagicLens + Qwen2-VL-2B bf16 + LoRA − (c) | +7.5 [+5.3, +9.9] | +11.7 [+9.5, +14.2] |
| Qwen2-VL-2B bf16 + LoRA chỉ VLM − (c) | +4.5 [+1.7, +7.3] | +6.6 [+3.7, +9.7] |
| MagicLens + Qwen2-VL-7B bf16 − MagicLens + Qwen2-VL-2B fp16 | +4.0 [+1.8, +6.1] | +5.2 [+3.1, +7.1] |
| MagicLens + Qwen2-VL-2B bf16 + LoRA − MagicLens + Qwen2-VL-2B fp16 | +3.8 [+1.5, +6.1] | +9.2 [+6.9, +11.5] |
| MagicLens + Qwen2-VL-2B bf16 + LoRA − MagicLens + Qwen2-VL-7B bf16 | -0.2 [-2.4, +2.0] | +4.1 [+2.0, +6.1] |
| MagicLens + Qwen2-VL-2B fp16 − (a) | +7.1 [+4.8, +9.3] | +13.3 [+10.7, +16.0] |
| MagicLens + Qwen2-VL-7B bf16 − (a) | +11.1 [+8.8, +13.5] | +18.5 [+15.5, +21.3] |
| MagicLens + Qwen2-VL-2B bf16 + LoRA − (a) | +10.8 [+8.4, +13.4] | +22.6 [+19.4, +25.7] |

## Từng VLM trên test

AUC: khả năng tách ảnh đích khỏi các ứng viên khác trong top-k đã chọn (0.5 là ngẫu nhiên), trung bình trên các truy vấn có ảnh đích trong top-k. Thời gian: giờ GPU đo trên Colab, không gồm tải và nạp mô hình; mỗi lần chạy có thể dùng GPU khác nhau.

| VLM | k | Truy vấn có ảnh đích trong top-k | AUC MagicLens | AUC VLM | AUC trộn | Tương quan | Giờ GPU | ms/cặp |
|---|---|---|---|---|---|---|---|---|
| Qwen2-VL-2B fp16 | 50 | 539 | 0.761 | 0.736 | 0.796 | 0.17 | 0.9 | 36 |
| Qwen2-VL-7B bf16 | 50 | 539 | 0.761 | 0.813 | 0.838 | 0.21 | 0.9 | 34 |
| Qwen2-VL-2B bf16 + LoRA | 50 | 539 | 0.761 | 0.841 | 0.870 | 0.13 | 0.3 | 13 |

## Ảnh đích lọt top-10 sau rerank, theo hạng của nó ở bước 1 (test)

Mỗi ô: tỉ lệ truy vấn của nhóm có ảnh đích trong top-10 sau khi trộn (nhóm 1–10: giữ lại được; các nhóm sau: đưa lên được).

| Hạng ở bước 1 | Số truy vấn | MagicLens + Qwen2-VL-2B fp16 | MagicLens + Qwen2-VL-7B bf16 | MagicLens + Qwen2-VL-2B bf16 + LoRA |
|---|---|---|---|---|
| 1–10 | 318 | 88.7% | 92.8% | 95.3% |
| 11–20 | 89 | 41.6% | 61.8% | 68.5% |
| 21–30 | 65 | 20.0% | 38.5% | 53.8% |
| 31–50 | 67 | 13.4% | 20.9% | 41.8% |

392 truy vấn test có ảnh đích ngoài top-50 của bước 1: không cấu hình rerank nào đưa được ảnh đích của chúng vào top-10.

## Ảnh truy vấn (test)

Kho ảnh chứa cả ảnh truy vấn, như cách đánh giá gốc, dù ảnh này không bao giờ là đáp án. Cột "lọc": bỏ ảnh truy vấn khỏi danh sách kết quả rồi tính lại.

| Cấu hình | Ảnh truy vấn trong top-10 | Ảnh truy vấn ở hạng 1 | R@1 | R@1 khi lọc | R@10 | R@10 khi lọc |
|---|---|---|---|---|---|---|
| (c) MagicLens tinh chỉnh (bước 1) | 121 | 28 | 11.4 | 11.5 | 34.2 | 34.3 |
| MagicLens + Qwen2-VL-2B fp16 | 33 | 2 | 15.1 | 15.1 | 36.6 | 36.6 |
| MagicLens + Qwen2-VL-7B bf16 | 34 | 8 | 19.1 | 19.3 | 41.8 | 41.8 |
| MagicLens + Qwen2-VL-2B bf16 + LoRA | 1 | 0 | 18.9 | 18.9 | 45.9 | 45.9 |

## Tương quan giữa điểm của các VLM (test)

Pearson giữa z(VLM) của hai VLM trên mọi cặp (truy vấn, ứng viên) trong top-50 của 931 truy vấn test. Để so: tương quan giữa z(MagicLens) và z(VLM) ở bảng "Từng VLM".

| | Qwen2-VL-2B fp16 | Qwen2-VL-7B bf16 | Qwen2-VL-2B bf16 + LoRA |
|---|---|---|---|
| Qwen2-VL-2B fp16 | 1.00 | 0.57 | 0.21 |
| Qwen2-VL-7B bf16 | 0.57 | 1.00 | 0.36 |
| Qwen2-VL-2B bf16 + LoRA | 0.21 | 0.36 | 1.00 |

## Kiểm tra kiểu số: Qwen2-VL-2B ở float16 và bfloat16

Cùng mô hình, cùng 200 truy vấn đầu (10000 cặp). Điểm float16 là của lần chạy Hướng 1, điểm bfloat16 của phần A, có thể trên GPU khác; nên phép so gồm mọi khác biệt khi tính toán, không riêng kiểu số.

- Tương quan Pearson giữa hai bộ điểm: 0.9930.
- Lệch tuyệt đối trung bình 0.070, lớn nhất 0.832 (độ lệch chuẩn của điểm: 0.85).
- Tương quan thứ hạng Spearman trong từng truy vấn, trung bình: 0.9604; cùng ứng viên đứng đầu theo VLM ở 69.0% truy vấn.
- Tiêu chí đặt trước khi chạy (Pearson ≥ 0.99 và Spearman trung bình ≥ 0.98): không đạt.

Chỉ số đánh giá trên cùng 200 truy vấn (101 dev, 99 test; R@10 của bước 1 trên các truy vấn này: 33.0). Bản bfloat16 được trộn với k và w đã chọn cho bản float16. Các VLM khác dùng k và w của chúng, để so độ lớn của các chênh lệch.

| VLM | k, w | AUC VLM | AUC trộn | R@1 trộn | R@10 trộn |
|---|---|---|---|---|---|
| Qwen2-VL-2B fp16 | 50, 1.5 | 0.740 | 0.806 | 11.0 | 37.5 |
| Qwen2-VL-2B bf16 | 50, 1.5 | 0.739 | 0.802 | 11.0 | 36.5 |
| Qwen2-VL-7B bf16 | 50, 2 | 0.806 | 0.842 | 17.0 | 39.0 |
| Qwen2-VL-2B bf16 + LoRA | 50, 2 | 0.841 | 0.875 | 21.0 | 45.0 |

- Từ Qwen2-VL-2B fp16 sang Qwen2-VL-2B bf16: 8 truy vấn đổi trạng thái (ảnh đích vào top-10: 3, ra khỏi top-10: 5); ΔR@10 -1.0 [-4.0, +1.5].
- Chênh AUC VLM so với Qwen2-VL-2B fp16: Qwen2-VL-2B bf16 -0.001, Qwen2-VL-7B bf16 +0.066, Qwen2-VL-2B bf16 + LoRA +0.101.

## Huấn luyện LoRA: kiểm tra dev của từng checkpoint

AUC và MRR của điểm VLM trên top-20 ứng viên của bước 1, ở các truy vấn dev có ảnh đích trong top-20 (vlm_lora_train.py). Bước 0 là mô hình chưa huấn luyện.

| Bước | Checkpoint | Loss train (trung bình từ lần kiểm tra trước) | AUC dev | MRR dev |
|---|---|---|---|---|
| 0 | chưa huấn luyện | — | 0.7303 | 0.3686 |
| 545 | ckpt-00545 | 0.9701 | 0.7776 | 0.4528 |
| 1090 | ckpt-01090 | 0.7678 | 0.7826 | 0.4581 |
| 1635 | ckpt-01635 | 0.6602 | 0.7824 | 0.4756 |
| 2178 | ckpt-02178 | 0.6473 | 0.7887 | 0.4683 |

Checkpoint được chọn: `ckpt-02178`.

Thời gian theo `train_log.jsonl`: 51 phút từ lần kiểm tra dev đầu đến checkpoint cuối, trong đó 5.1 phút cho 5 lần kiểm tra dev; không gồm tải và nạp mô hình.
