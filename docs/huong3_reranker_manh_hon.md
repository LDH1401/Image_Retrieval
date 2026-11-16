# Hướng 3: reranker mạnh hơn

Hướng 1 cho thấy VLM Qwen2-VL-2B zero-shot đã trở thành điểm yếu của hệ thống:

- **VLM tách ảnh đích kém hơn bước 1.** Trên ứng viên của MagicLens tinh chỉnh, AUC của VLM là 0.736, còn của MagicLens là 0.761.
- **Còn nhiều chỗ để cải thiện.** Ảnh đích nằm trong top-50 ở 57.9% truy vấn test, nhưng sau rerank chỉ 36.6% có ảnh đích trong top-10.

Hướng 3 giữ nguyên tập ứng viên đó (top-50 của MagicLens tinh chỉnh) và **chỉ thay VLM**. Đây là mục 6 của tài liệu đề xuất cải tiến (`DeXuat_CaiTien_MagicLens.docx`), gồm hai thí nghiệm:

- **6.1, VLM lớn hơn:** Qwen2-VL-7B, vẫn zero-shot.
- **6.2, VLM được dạy đúng tác vụ:** Qwen2-VL-2B tinh chỉnh LoRA trên tập train của FashionIQ dress.

Kèm theo là một phép kiểm tra nhỏ: chấm lại bản 2B ở bfloat16 để biết riêng kiểu số làm điểm thay đổi bao nhiêu.

**Kết quả (mục 5).** Cả hai thí nghiệm đều cải thiện rõ so với Hướng 1. Số dưới đây là R@10 trên 931 truy vấn test:
- 2B zero-shot (Hướng 1) đạt 36.6.
- 7B đạt 41.8.
- 2B + LoRA đạt 45.9, cao nhất. Bản này hơn cả 7B: +4.1 [+2.0, +6.1].

LoRA chỉ học 51 phút trên Colab. Đổi kiểu số sang bfloat16 không phải nguyên nhân của mức tăng.

## 1. Các cấu hình so sánh

Mọi cấu hình dùng cùng kho ảnh (3653 ảnh val dress), cùng 1861 truy vấn, cùng cách chia dev 930 / test 931 (seed 0) và cùng tập ứng viên `vlm_rerank_ft/candidates.json` của Hướng 1. Số trong bảng là R@10 trên test.

| Ký hiệu | VLM chấm top-50 | Kiểu số | VLM học trên FashionIQ | R@10 test |
|---|---|---|---|---|
| (c) | Không rerank | — | — | 34.2 |
| (e) | Qwen2-VL-2B | float16 | Không | 36.6 (Hướng 1) |
| (f) | Qwen2-VL-7B | bfloat16 | Không | 41.8 |
| (g) | Qwen2-VL-2B + LoRA | bfloat16 | Có (5441 truy vấn train) | **45.9** |

Mỗi VLM có hai cách dùng: xếp lại chỉ theo điểm VLM, hoặc trộn z(MagicLens) + w · z(VLM). Cũng như Hướng 1, k và w được chọn trên dev.

## 2. Những gì có trong repo

| Tệp | Việc làm |
|---|---|
| `vlm_rerank.py` | Thêm `--adapter`: chấm điểm bằng mô hình gắn checkpoint LoRA (gộp vào trọng số). Tên và mã băm của adapter được ghi vào `.meta.json`, nên không thể chạy tiếp nhầm sang adapter khác. |
| `vlm_rerank_eval.py` | Tên tệp kết quả có thêm tên VLM khi VLM khác bản gốc (2B, float16, không LoRA), để kết quả của 7B và LoRA không đè lên Hướng 1. Câu mô tả ghi đúng khi VLM đã được tinh chỉnh. Kết quả cũ không đổi. |
| `vlm_lora_prepare.py` | **Mới.** Tạo dữ liệu huấn luyện LoRA trên máy (mục 3) và đóng gói `vlm_lora/lora_train_bundle.zip`. |
| `vlm_lora_train.py` | **Mới.** Huấn luyện LoRA trên Colab, chọn checkpoint theo dev, chạy tiếp được khi bị ngắt. |
| `huong3_colab.ipynb` | **Mới.** Notebook Colab gồm bốn phần A–D (mục 4). |
| `huong3_tong_hop.py` | **Mới.** Bảng so sánh 2B, 7B và 2B + LoRA trên cùng tập ứng viên, kèm bootstrap ghép cặp, cùng các phân tích ở mục 5. |

`vlm_lora/lora_train_bundle.zip` (847 MB) chỉ nằm trên máy; git bỏ qua tệp này. `vlm_lora/candidates_train.json` (5 MB) được đưa lên git để ghi lại đúng dữ liệu đã dùng.

## 3. Dữ liệu và cấu hình huấn luyện LoRA

### Dữ liệu (đã tạo trên máy)

```bash
.venv/bin/python vlm_lora_prepare.py --subtask dress --out_dir ./vlm_lora
```

Lệnh này đã chạy trên máy, mất 1.7 phút. Nó làm các việc sau:

1. **Xếp hạng kho train.** Dùng MagicLens **zero-shot** (checkpoint gốc) xếp hạng kho train (10 878 ảnh) cho từng truy vấn train (5441 truy vấn), và giữ top-50.
   - Không dùng bản tinh chỉnh, vì bản đó đã học trên chính tập train này: nó tìm ảnh đích quá dễ, nên danh sách ứng viên khác hẳn những gì gặp lúc test.
   - Trên kho train, bản zero-shot đạt R@10 14.3, R@50 29.6. Số thấp hơn trên val vì kho lớn gấp 3.
2. **Đóng gói.** Gói gồm `candidates_train.json`, hai script `vlm_rerank.py` và `vlm_lora_train.py`, và 10 596 ảnh cần dùng.
3. **Không dùng truy vấn hay ảnh nào của tập val.** Tập train và tập val của dress không có ảnh chung.

Mỗi truy vấn train cho một nhóm 6 cặp (ảnh truy vấn, ảnh ứng viên, câu chỉ dẫn), với đúng câu hỏi Yes/No dùng khi chấm điểm:

| Ứng viên | Nhãn | Lý do |
|---|---|---|
| Ảnh đích | Yes | Mẫu dương |
| 4 ảnh rút ngẫu nhiên từ hạng 6–50 của MagicLens, rút lại mỗi epoch | No | Mẫu âm khó: những chiếc váy giống nhau mà hệ thống hay nhầm. Bỏ 5 hạng đầu vì FashionIQ chỉ gắn một ảnh đích cho mỗi truy vấn, trong khi các ảnh đứng đầu thường cũng khớp với câu mô tả (mẫu âm giả). |
| Chính ảnh truy vấn | No | Ảnh chưa được sửa thì không thể là kết quả. Ở tập train, ảnh này chỉ lọt top-50 của 6.3% truy vấn. Nhưng trong ứng viên val của bản tinh chỉnh, nó nằm trong top-10 ở 121 truy vấn test, nên cần dạy VLM trả lời "No" cho nó. |

Một epoch có 5441 × 6 = 32 646 cặp.

### Cấu hình huấn luyện (mặc định của `vlm_lora_train.py`)

Các giá trị theo Bảng 10 của tài liệu đề xuất. Đây là điểm khởi đầu hợp lý, chưa được tinh chỉnh.

| Thành phần | Giá trị |
|---|---|
| Đại lượng được học | s = logit("Yes") − logit("No") ở token trả lời đầu tiên. Script dùng lại lớp `Scorer` của `vlm_rerank.py`, nên đây đúng là điểm dùng khi chấm (cùng câu hỏi, chat template, cỡ ảnh 448 × 448). |
| Phần được học | LoRA trên các phép chiếu attention q, k, v, o của phần ngôn ngữ, khoảng 4.4 triệu tham số. Encoder ảnh, embedding và lớp đầu ra đóng băng. |
| Tham số LoRA | r = 16, alpha = 32, dropout 0.05 |
| Hàm mất mát | Cross-entropy nhị phân trên s. Ảnh đích được nhân trọng số 5 để cân bằng với 5 mẫu âm. Biến thể: `--loss listwise`, softmax trên 6 cặp của nhóm. |
| Tối ưu | AdamW, lr 1e-4 (5% số bước đầu tăng dần, sau đó giảm tuyến tính về 0), cắt gradient ở 1.0, bfloat16 |
| Batch | 5 truy vấn mỗi bước, tức 30 cặp. 2 epoch = 2178 bước. |
| Chọn checkpoint | Trước khi học và sau mỗi nửa epoch (các bước 545, 1090, 1635, 2178), chấm 200 truy vấn **dev** × top-20 ứng viên của Hướng 1. Chỉ lấy các truy vấn dev có ảnh đích trong top-20. Ghi AUC (khả năng tách ảnh đích khỏi 19 ứng viên còn lại) và MRR. Checkpoint có AUC dev cao nhất được ghi vào `best.json`. |

Vì encoder ảnh đóng băng, đặc trưng của mỗi ảnh chỉ tính một lần rồi giữ trong RAM, giống cách `vlm_rerank.py` làm khi chấm. Với 2B, phần này tốn khoảng 6 GB RAM, dư sức với runtime L4 hoặc A100 của Colab.

## 4. Các bước

### Bước 1. Tải lên Google Drive

1. Mở thư mục **`MyDrive/magiclens_vlm_ft/`** của Hướng 1. Kiểm tra trong đó còn `bundle.zip`, `vlm_scores.jsonl` và `vlm_scores.jsonl.meta.json`. Phần A cần điểm 2B của Hướng 1 để so sánh.
2. Tải `vlm_lora/lora_train_bundle.zip` (847 MB) vào **cùng thư mục đó**, không giải nén. Chỉ phần C và D cần tệp này; phần A và B chạy được ngay cả khi chưa tải.

### Bước 2. Mở notebook trên Colab

1. Vào [colab.research.google.com](https://colab.research.google.com), chọn **File → Upload notebook**, rồi chọn `huong3_colab.ipynb` trong repo.
2. Vào **Runtime → Change runtime type**, chọn **A100** (nhanh nhất) hoặc **L4**. **T4 không dùng được:** T4 không có bfloat16, và bản 7B (khoảng 17 GB) không vừa 15 GB bộ nhớ của nó.
3. Chạy ô 1 đến ô 5:
   - **Ô 1** in tên GPU và phải báo `bfloat16: True`.
   - **Ô 3** phải in `OK`.
   - **Ô 5** phải in `Ứng viên của bước 1: magic_lens_clip_base_fiq_dress_ft.pkl` và cho biết phần C, D đã sẵn sàng chưa.

Mỗi khi Colab khởi động lại (bị ngắt, mở lại notebook), luôn chạy lại ô 1–5 trước, rồi mới chạy ô của phần đang làm.

### Bước 3. Chạy bốn phần

Các phần độc lập với nhau, trừ D cần C xong trước. Có thể chạy mỗi phần trong một phiên khác nhau. Thứ tự đề xuất: A → B → C → D.

**Phần A: kiểm tra kiểu số (khoảng 10 phút).** Chấm lại bản 2B ở bfloat16 trên 200 truy vấn đầu, rồi so với điểm float16 của Hướng 1. Ô in ra tương quan Pearson, mức lệch, tương quan thứ hạng trong từng truy vấn, và một dòng KẾT LUẬN.
- Nếu là "bfloat16 gần như không đổi điểm", chênh lệch giữa 2B và 7B ở phần B đến từ kích thước mô hình.
- Nếu không, cần ghi chú điều này khi so sánh.

**Phần B: Qwen2-VL-7B (ước tính 3–4 giờ trên L4, ít hơn trên A100).**
- **Ô B1** tải mô hình (khoảng 17 GB, vài phút) và chạy thử 3 truy vấn theo hai cách. Ô phải in **OK**.
- **Ô B2** chấm đủ 93 050 cặp, ghi lên Drive sau mỗi truy vấn. Bị ngắt thì chạy lại ô 1–5 rồi ô B2; script làm tiếp từ chỗ dừng.
- Muốn nhanh hơn thì đặt `TOP_K_7B = 30` ở ô 3 **trước khi** chạy B2. Số cặp giảm 40%, nhưng trần R@10 giảm từ 57.9 xuống 50.7.
- Báo hết bộ nhớ GPU thì giảm `BATCH_7B`.

**Phần C: huấn luyện LoRA (ước tính 2–2.5 giờ).** Ô in mỗi lần kiểm tra dev một dòng dạng `step 545/2178: dev AUC …, MRR …, train loss …`.
- **Dòng `step 0`** là mô hình chưa học, tức mốc để so.
- **LoRA có tác dụng** nếu AUC dev ở các checkpoint vượt rõ mốc này.
- **Dấu hiệu học quá:** AUC dev tăng rồi giảm ở cuối. Checkpoint được chọn vẫn là bản có AUC dev cao nhất, nên không cần can thiệp.
- **Khi kết thúc,** ô in bảng các checkpoint và tên checkpoint tốt nhất (ghi trong `lora_2b/best.json` trên Drive).
- **Bị ngắt:** chạy lại ô 1–5 rồi ô C. Script học tiếp từ checkpoint gần nhất, mất tối đa nửa epoch, và kết quả vẫn trùng một lần chạy liền mạch.
- **Báo hết bộ nhớ GPU:** thêm `--grad_ckpt` vào cuối lệnh trong ô. Chạy chậm hơn khoảng 30% nhưng kết quả không đổi.

**Phần D: chấm bằng LoRA (ước tính khoảng 1 giờ).** Ô đọc `best.json`, rồi chấm đúng 93 050 cặp bằng 2B + checkpoint đó, ở bfloat16 như lúc huấn luyện. Bị ngắt thì chạy lại ô 1–5 rồi ô D.

Các thời gian trên là **ước tính** làm trước khi chạy, dựa trên tốc độ đã đo ở Hướng 1 (2B: 0.9 giờ cho 93 050 cặp). Lần chạy thật nhanh hơn: A 3 phút, B 0.9 giờ, C 51 phút, D 0.34 giờ (mục 5.5). Ô "Kiểm tra tiến độ" ở cuối notebook in thời gian thật của từng phần.

Colab tính phí theo thời gian GPU được kết nối, kể cả khi không chạy gì. Xong việc thì chọn **Runtime → Disconnect and delete runtime**.

### Bước 4. Tải kết quả về máy

Chạy hai ô cuối của notebook (hoặc tải bằng giao diện Drive), rồi chép:

| Tệp trên Drive (`MyDrive/magiclens_vlm_ft/`) | Chép vào |
|---|---|
| `vlm_scores_2b_bf16_check.jsonl` và `.meta.json` đi kèm | `vlm_rerank_ft/` |
| `vlm_scores_7b.jsonl` và `.meta.json` đi kèm | `vlm_rerank_ft/` |
| `vlm_scores_2b_lora.jsonl` và `.meta.json` đi kèm | `vlm_rerank_ft/` |
| `lora_2b/best.json` và `lora_2b/train_log.jsonl` | `vlm_lora/` |

Không cần tải các thư mục `lora_2b/ckpt-…`: điểm đã chấm xong trên Colab.

### Bước 5. Đánh giá và tổng hợp (máy, vài giây)

```bash
.venv/bin/python vlm_rerank_eval.py --candidates vlm_rerank_ft/candidates.json \
    --scores vlm_rerank_ft/vlm_scores_7b.jsonl --output ./output/huong3
.venv/bin/python vlm_rerank_eval.py --candidates vlm_rerank_ft/candidates.json \
    --scores vlm_rerank_ft/vlm_scores_2b_lora.jsonl --output ./output/huong3
.venv/bin/python huong3_tong_hop.py
```

- **Hai lệnh đầu** chọn k và w trên dev cho từng VLM, rồi ghi kết quả vào `output/huong3/`. Tên tệp mang tên VLM, ví dụ `..._qwen2-vl-7b-instruct_bf16.md`.
- **Lệnh cuối** ghi `output/huong3/huong3_tong_hop.md`, gồm:
  - Bảng (a), (c), (e), (f), (g), cả bản "chỉ VLM" lẫn bản trộn.
  - Các phép so sánh bootstrap trên test (ΔR@1 và ΔR@10), kể cả so với MagicLens zero-shot (a).
  - AUC, tương quan và thời gian của từng VLM.
  - Tỉ lệ ảnh đích lọt top-10 theo hạng của nó ở bước 1, kết quả khi lọc ảnh truy vấn, và tương quan giữa điểm của các VLM.
  - Kết quả phần A, gồm cả ảnh hưởng của kiểu số lên AUC và R@10.
  - Bảng kiểm tra dev của từng checkpoint LoRA và thời gian huấn luyện.

Phần nào chưa chạy thì được ghi là "chưa có"; có thể tổng hợp sau khi xong từng phần.

## 5. Kết quả

Cả bốn phần đã chạy xong trên Colab ngày 16/11/2026. Bảng đầy đủ nằm ở [output/huong3/huong3_tong_hop.md](../output/huong3/huong3_tong_hop.md). Kết quả riêng của từng VLM nằm cùng thư mục, tên tệp mang tên VLM.

### 5.1. Kết quả chính (test, 931 truy vấn)

| Cấu hình | R@1 | R@5 | R@10 | R@50 | ΔR@10 so với (c) |
|---|---|---|---|---|---|
| (a) MagicLens zero-shot | 8.1 | 17.8 | 23.3 | 42.1 | |
| (c) MagicLens tinh chỉnh (bước 1) | 11.4 | 25.7 | 34.2 | 57.9 | |
| (e) (c) + Qwen2-VL-2B, top-50, w = 1.5 (Hướng 1) | 15.1 | 29.8 | 36.6 | 57.9 | +2.5 [+0.5, +4.4] |
| (f) (c) + Qwen2-VL-7B, top-50, w = 2 | **19.1** | 34.2 | 41.8 | 57.9 | +7.6 [+5.5, +9.8] |
| (g) (c) + Qwen2-VL-2B + LoRA, top-50, w = 2 | 18.9 | **37.2** | **45.9** | 57.9 | **+11.7 [+9.5, +14.2]** |

Cách đọc bảng:
- Khoảng trong ngoặc vuông là khoảng tin cậy 95%, tính bằng bootstrap ghép cặp 2000 lần.
- k và w của mỗi VLM được chọn trên dev. R_avg dev: (e) 48.2, (f) 49.8, (g) 51.4.
- Rerank chỉ xếp lại top-50, nên R@50 không đổi và R@10 không thể vượt 57.9.

So sánh trực tiếp giữa các VLM:

| So sánh | ΔR@1 | ΔR@10 |
|---|---|---|
| (f) − (e): VLM lớn hơn | +4.0 [+1.8, +6.1] | +5.2 [+3.1, +7.1] |
| (g) − (e): VLM được dạy đúng tác vụ | +3.8 [+1.5, +6.1] | +9.2 [+6.9, +11.5] |
| (g) − (f): 2B + LoRA so với 7B | −0.2 [−2.4, +2.0] | +4.1 [+2.0, +6.1] |
| (g) − (a): so với MagicLens zero-shot | +10.8 [+8.4, +13.4] | +22.6 [+19.4, +25.7] |

### 5.2. Nhận xét

- **Cả hai cách đều có ích, LoRA tốt nhất.**
  - So với 2B: 7B tăng R@10 thêm 5.2 điểm, LoRA tăng thêm 9.2 điểm.
  - 2B + LoRA hơn 7B ở R@10 (+4.1) và R@5 (37.2 so với 34.2). R@1 của hai bản ngang nhau (18.9 và 19.1).
  - So với MagicLens zero-shot, R@10 gần gấp đôi: từ 23.3 lên 45.9.
- **VLM nay tách ảnh đích tốt hơn MagicLens.**
  - AUC trên 539 truy vấn test có ảnh đích trong top-50: MagicLens 0.761, VLM 2B 0.736, 7B 0.813, 2B + LoRA 0.841.
  - Chỉ xếp lại theo VLM cũng đã hơn (c): 7B +4.6 [+1.9, +7.5], LoRA +6.6 [+3.7, +9.7] R@10. 2B của Hướng 1 chỉ hơn +1.1, không có ý nghĩa thống kê.
  - Trọng số w chọn trên dev cũng tăng, từ 1.5 lên 2.
- **Trộn vẫn tốt hơn chỉ dùng VLM.**
  - Điểm MagicLens và điểm VLM tương quan thấp: 0.21 với 7B, 0.13 với LoRA.
  - Trộn tốt hơn chỉ dùng VLM: +3.0 [+1.2, +4.8] với 7B, +5.2 [+3.1, +7.2] với LoRA.
  - AUC sau trộn: 0.838 với 7B, 0.870 với LoRA.
- **Reranker mạnh hơn kéo được ảnh đích từ hạng sâu hơn.** Bảng dưới là tỉ lệ truy vấn có ảnh đích trong top-10 sau khi trộn, chia theo hạng của ảnh đích ở bước 1:

  | Hạng ở bước 1 | Số truy vấn | 2B | 7B | 2B + LoRA |
  |---|---|---|---|---|
  | 1–10 (giữ lại) | 318 | 88.7% | 92.8% | 95.3% |
  | 11–20 | 89 | 41.6% | 61.8% | 68.5% |
  | 21–30 | 65 | 20.0% | 38.5% | 53.8% |
  | 31–50 | 67 | 13.4% | 20.9% | 41.8% |

  - Số truy vấn có ảnh đích được đưa vào / bị đẩy ra khỏi top-10: 2B 59 / 36, 7B 94 / 23, LoRA 124 / 15.
  - Với LoRA, rerank càng sâu càng tốt: R@10 test là 40.2 ở top-20, 43.8 ở top-30, 45.9 ở top-50. Với 2B, R@10 gần như không đổi (36.4–36.6).
- **LoRA không thắng nhờ loại ảnh truy vấn.**
  - LoRA được dạy trả lời "No" cho chính ảnh truy vấn. Sau rerank, ảnh này chỉ còn trong top-10 ở 1 truy vấn test (bước 1: 121; 2B: 33; 7B: 34).
  - Nhưng nếu lọc hẳn ảnh truy vấn khỏi kết quả của mọi cấu hình, R@10 gần như không đổi: (c) 34.3, 2B 36.6, 7B 41.8, LoRA 45.9.
  - Vậy mức tăng đến từ việc xếp ảnh đích lên cao hơn, không phải từ việc loại ảnh truy vấn.
- **Phần còn lại bị giới hạn bởi bước 1.** 392 truy vấn test (42.1%) có ảnh đích nằm ngoài top-50 của bước 1, nên không reranker nào cứu được. LoRA đạt 45.9 trên mức trần 57.9.
- **Ba VLM chấm khá khác nhau.**
  - Tương quan giữa điểm z của các VLM trên test: 2B–7B 0.57, 7B–LoRA 0.36, 2B–LoRA 0.21.
  - Tương quan thấp giữa 2B và LoRA cho thấy LoRA đã làm điểm của 2B thay đổi nhiều.
  - Tương quan 0.36 giữa 7B và LoRA gợi ý ghép cả hai có thể tốt thêm. Chưa thử.

### 5.3. Kiểm tra kiểu số (phần A)

Phần A chấm lại Qwen2-VL-2B ở bfloat16 trên 200 truy vấn đầu (101 dev, 99 test; 10 000 cặp), rồi so với điểm float16 của Hướng 1. Hai lần chạy có thể dùng GPU khác nhau, nên phép so gồm mọi khác biệt khi tính toán, không riêng kiểu số.

- **Điểm số:**
  - Tương quan Pearson 0.9930.
  - Lệch tuyệt đối trung bình 0.070, trong khi độ lệch chuẩn của điểm là 0.85.
  - Tương quan Spearman trong từng truy vấn trung bình 0.960. Cùng ứng viên đứng đầu ở 69% truy vấn.
  - Tiêu chí đặt trước khi chạy (Pearson ≥ 0.99 và Spearman ≥ 0.98) **không đạt**, vì thứ tự của các ứng viên có điểm sát nhau bị đảo.
- **Chỉ số đánh giá trên cùng 200 truy vấn:**

  | VLM | AUC VLM | AUC trộn | R@10 trộn |
  |---|---|---|---|
  | 2B float16 (Hướng 1) | 0.740 | 0.806 | 37.5 |
  | 2B bfloat16 | 0.739 | 0.802 | 36.5 |
  | 7B bfloat16 | 0.806 | 0.842 | 39.0 |
  | 2B + LoRA bfloat16 | 0.841 | 0.875 | 45.0 |

  Đổi sang bfloat16 làm 8 truy vấn đổi trạng thái: 3 có ảnh đích vào top-10, 5 có ảnh đích ra khỏi top-10. ΔR@10 là −1.0 [−4.0, +1.5].
- **Kết luận:**
  - bfloat16 thêm nhiễu nhỏ ở từng truy vấn, nhưng không làm VLM tốt lên hay kém đi một cách có hệ thống.
  - AUC chỉ đổi −0.001, trong khi đổi sang 7B tăng +0.066 và thêm LoRA tăng +0.101.
  - Nếu có thiên lệch, bfloat16 hơi bất lợi. Vậy mức tăng của (f) và (g), đều chạy bfloat16, không do kiểu số tạo ra.

### 5.4. Huấn luyện LoRA (phần C)

| Bước | Loss train | AUC dev | MRR dev |
|---|---|---|---|
| 0 (chưa học) | — | 0.730 | 0.369 |
| 545 (nửa epoch) | 0.970 | 0.778 | 0.453 |
| 1090 (1 epoch) | 0.768 | 0.783 | 0.458 |
| 1635 | 0.660 | 0.782 | 0.476 |
| 2178 (2 epoch) | 0.647 | **0.789** | 0.468 |

- AUC dev tăng mạnh nhất trong nửa epoch đầu, sau đó tăng chậm.
- Checkpoint cuối có AUC dev cao nhất nên được chọn, đúng tiêu chí đặt trước. MRR dev cao nhất ở bước 1635, nhưng MRR không dùng để chọn.
- Huấn luyện 2 epoch mất 51 phút, gồm 5.1 phút cho 5 lần kiểm tra dev.
- Các tham số vẫn là mặc định, chưa được tinh chỉnh (mục 6).

### 5.5. Chi phí

| Phần | Việc | Thời gian GPU | ms/cặp |
|---|---|---|---|
| A | 2B bfloat16, 200 truy vấn | 3 phút | — |
| B | 7B, 93 050 cặp | 0.9 giờ | 34 |
| C | Huấn luyện LoRA, 2 epoch | 51 phút | — |
| D | 2B + LoRA, 93 050 cặp | 0.34 giờ | 13 |

- Tổng cộng khoảng 2.1 giờ GPU, chưa gồm tải và nạp mô hình.
- Để so: 2B float16 ở Hướng 1 mất 0.9 giờ (36 ms/cặp).
- Loại GPU của từng lần chạy không được ghi lại, nên không so tốc độ giữa các lần chạy được.

### 5.6. So sánh công bằng

Theo mục 7.2 của tài liệu đề xuất:
- Mọi cấu hình của Hướng 3 đều là hệ có giám sát, vì bước 1 đã tinh chỉnh trên FashionIQ train. (g) còn học thêm trên chính tập train đó.
- Vì vậy không đặt các số này cạnh bảng zero-shot của bài báo. Xem [ket_qua.md](ket_qua.md), mục 3.
- Mọi lựa chọn (k, w, checkpoint LoRA) chỉ dựa trên dev. Mỗi cấu hình chỉ được chấm trên test một lần.
- LoRA chỉ học trên ảnh và truy vấn của tập train. Tập train và tập val của dress không có ảnh chung.

## 6. Thay đổi cấu hình (nếu còn thời gian GPU)

Các tham số đáng thử trong `vlm_lora_train.py`:
- `--skip`: số hạng đầu bỏ qua khi rút mẫu âm (mặc định 5).
- `--n_neg`: số mẫu âm mỗi truy vấn (mặc định 4).
- `--loss listwise`: biến thể của hàm mất mát.
- `--epochs`.

Mỗi lần đổi cần một thư mục `--out_dir` mới. Nếu dùng lại thư mục cũ, script sẽ từ chối vì cấu hình khác lần chạy đang có trong đó. Chỉ so các lần thử bằng AUC dev (`best.json`). Chỉ checkpoint cuối cùng được chọn mới được chấm trên test (phần D), để số test không bị dùng vào việc chọn cấu hình.

## 7. Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| Ô 1 báo không có bfloat16 | Runtime T4 | Đổi sang A100 hoặc L4 |
| `CUDA out of memory` ở phần B | Batch quá lớn với 7B | Giảm `BATCH_7B` ở ô 3 (ví dụ 4), chạy lại ô 3 rồi ô B2 |
| `CUDA out of memory` ở phần C | Thiếu bộ nhớ khi tính gradient | Thêm `--grad_ckpt` vào cuối lệnh trong ô C |
| `... was made with other settings (...)` | Đã đổi `TOP_K_7B`, kiểu số hoặc adapter giữa chừng | Đặt lại như lúc bắt đầu, hoặc đổi tên tệp `OUT_…` để chạy lại từ đầu |
| `... holds a run with other settings (...)` | Đổi tham số huấn luyện mà vẫn dùng thư mục `lora_2b` cũ | Giữ nguyên tham số, hoặc đặt `LORA_DIR` mới |
| Phần D báo chưa có `best.json` | Phần C chưa chạy xong | Chạy tiếp ô C cho đến khi in "Checkpoint tốt nhất trên dev" |
| Phiên Colab bị dừng vì hết RAM ở phần C | Runtime ít RAM | Thêm `--no_vision_cache` vào lệnh trong ô C (chậm hơn) |
| `vlm_rerank_eval.py` báo `Only N/1861 queries are scored` | Tệp điểm chưa đủ, hoặc là tệp của phần A (chỉ 200 truy vấn) | Chạy tiếp phần tương ứng trên Colab. Không đánh giá tệp của phần A bằng `vlm_rerank_eval.py`; `huong3_tong_hop.py` tự đọc tệp đó. |

## 8. Đã kiểm thử

Mã PyTorch được kiểm thử trên CPU với một bản Qwen2-VL **tí hon, khởi tạo ngẫu nhiên**. Bản này có cùng kiến trúc, tokenizer và bộ xử lý ảnh của Qwen2-VL-2B, nhưng chỉ có 2 lớp. Thư viện dùng đúng bản trên Colab: transformers 4.57.6, peft 0.17.1.

- **Không đổi kết quả cũ:** `vlm_rerank_eval.py` đã sửa cho ra tệp kết quả của Hướng 1 trùng từng byte với bản đã commit. Với lần rerank zero-shot, kết quả trùng với bản script ở commit trước.
- **`vlm_rerank.py`:**
  - Hai cách chạy (dùng lại đặc trưng ảnh và mã hóa lại mọi ảnh) cho điểm giống hệt nhau.
  - Khi không dùng `--adapter`, tệp `.meta.json` giữ đúng định dạng cũ.
- **Huấn luyện LoRA:**
  - Gradient chỉ đến 16 tensor LoRA của phần ngôn ngữ; encoder ảnh không bị gắn LoRA.
  - Với một tập mẫu âm cố định, cả hai hàm mất mát học thuộc được: loss về sát mức tối thiểu lý thuyết.
  - Hai lần chạy cùng seed cho trọng số trùng nhau.
  - Một lần chạy bị ngắt sau checkpoint đầu rồi chạy tiếp cho trọng số trùng lần chạy liền mạch (lệch tối đa 7 × 10⁻⁹, tức sai số làm tròn).
- **Chấm bằng adapter:**
  - Điểm của `vlm_rerank.py --adapter` (LoRA đã gộp) khớp điểm của mô hình lúc huấn luyện (LoRA chưa gộp), lệch tối đa 4.7 × 10⁻⁵, đúng bằng mức làm tròn khi ghi điểm.
  - Chạy tiếp với adapter khác bị từ chối.
- **Tổng hợp:** `vlm_rerank_eval.py` và `huong3_tong_hop.py` chạy với điểm 7B và LoRA giả lập. Hàng 2B khớp Hướng 1: 36.6 và +2.5 [+0.5, +4.4].
- **Notebook:** toàn bộ notebook được mô phỏng trên CPU (Drive giả, mô hình tí hon). Các phần A, B, C, D, ô tiến độ và ô tải về chạy đúng thứ tự, đọc đúng tệp của nhau.
- **`vlm_lora_prepare.py`:** chạy hai lần cho cùng `candidates_train.json`. Script trong `lora_train_bundle.zip` trùng với script trong repo.

**Chạy thật:** ngày 16/11/2026, cả bốn phần A–D chạy xong trên Colab với Qwen2-VL-2B và 7B thật, không cần sửa mã. Kết quả ở mục 5. Các tham số huấn luyện vẫn là mặc định lấy từ tài liệu đề xuất, chưa được tinh chỉnh.
