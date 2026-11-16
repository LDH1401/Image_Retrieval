# Benchmark của MagicLens và cách tái hiện

Tài liệu này tổng hợp các benchmark mà bài báo [*MagicLens: Self-Supervised Image Retrieval with Open-Ended Instructions*](https://arxiv.org/abs/2403.19651) (ICML 2024) báo cáo, cho biết benchmark nào chạy lại được bằng mã trong repo, và hướng dẫn từng bước để tái hiện chúng trên Google Colab (GPU A100).

Số liệu được lấy từ Bảng 1, 2, 4, 12, 14 và 15 của bài báo, và từ bảng kết quả CIRCO trong [README](../README.md). Tất cả là kết quả zero-shot: mô hình không được huấn luyện trên dữ liệu của các benchmark này.

## 1. Tổng quan

Bài báo báo cáo trên tám benchmark, chia thành hai nhóm. Bảng dưới chỉ ghi các bản dùng backbone CLIP, vì repo chỉ công bố checkpoint CLIP.

**Truy xuất từ (ảnh, chỉ dẫn) ra ảnh** (Bảng 1, 2):

| Nhiệm vụ | Benchmark | Số truy vấn | Kho ảnh | Chỉ số | MagicLens-B | MagicLens-L | Repo chạy được |
|---|---|---|---|---|---|---|---|
| Truy xuất ảnh có ghép | FashionIQ (FIQ) | 2005¹ | 5179¹ | R@10 | 26.3 | 30.7 | Có |
| Truy xuất ảnh có ghép | CIRR | 4148 | 2316 | R@1 / Rs@1 | 27.0 / 66.7 | 30.1 / 68.1 | Không |
| Truy xuất ảnh có ghép | CIRCO | 800 | 123 403 | mAP@5 | 23.1 | 29.6 | Có (nộp máy chủ) |
| Chuyển miền | DTIN | 10 000 | 16 983 | R@10 | 28.3 | 41.5 | Có |
| Độ tương đồng có điều kiện | GeneCIS | 2008 | tập con 13.8 ảnh² | Rs@1 | 15.0 | 16.3 | Không |

¹ Trung bình trên ba danh mục dress, shirt, toptee.
² Mỗi truy vấn tìm trong một tập con riêng, trung bình 13.8 ảnh.

**Truy xuất ảnh từ ảnh phác thảo** (Bảng 4; mọi truy vấn dùng chung chỉ dẫn `"find a natural image of it"`):

| Benchmark | Chỉ số | MagicLens-B | MagicLens-L | Repo chạy được |
|---|---|---|---|---|
| TU-Berlin | mAP / P@100 | 45.9 / 57.9 | 62.9 / 73.1 | Không |
| Sketchy | mAP@200 / P@200 | 49.3 / 60.6 | 68.2 / 75.8 | Không |
| QuickDraw | mAP / P@200 | 10.1 / 14.0 | 15.1 / 20.4 | Không |

**Những phần không tái hiện được bằng repo:**
- **CIRR, GeneCIS và ba benchmark ảnh phác thảo:** repo không có mã nạp dữ liệu cho các benchmark này. Muốn chạy thì phải tự viết hàm `build_*_dataset` theo mẫu trong [data_utils.py](../data_utils.py).
- **Các bản dùng backbone CoCa:** đây là các bản tốt nhất trong bài báo, ví dụ FIQ R@10 đạt 35.2 (Base) và 38.0 (Large), nhưng không có checkpoint công bố.
- **Checkpoint công bố là bản đã chuyển đổi định dạng,** nên kết quả lệch nhẹ so với bài báo. README chỉ công bố mức lệch trên CIRCO (mục 2.2).

## 2. Số liệu chi tiết của các benchmark chạy được

### 2.1. FashionIQ (Bảng 12, R@10 / R@50)

| Mô hình | dress | shirt | toptee | Trung bình |
|---|---|---|---|---|
| MagicLens-B (CLIP) | 21.5 / 41.3 | 27.3 / 48.8 | 30.2 / 52.3 | 26.3 / 47.4 |
| MagicLens-L (CLIP) | 25.5 / 46.1 | 32.7 / 53.8 | 34.0 / 57.7 | 30.7 / 52.5 |

Đánh giá trên tập val. Kho ảnh và số truy vấn của từng danh mục: dress 3817 ảnh / 2017 truy vấn, shirt 6346 / 2038, toptee 5373 / 1961.

### 2.2. CIRCO (Bảng 14 và README, mAP@K)

| Mô hình | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|---|---|---|---|---|
| MagicLens-B, bài báo | 23.1 | 23.8 | 25.8 | 26.7 |
| MagicLens-B, checkpoint chuyển đổi | 22.3 | 23.2 | 25.0 | 26.0 |
| MagicLens-L, bài báo | 29.6 | 30.8 | 33.4 | 34.4 |
| MagicLens-L, checkpoint chuyển đổi | 29.5 | 30.8 | 33.2 | 34.3 |

Đánh giá trên tập test (800 truy vấn), nhãn không công khai. Khi tái hiện, nên so với các dòng "checkpoint chuyển đổi".

### 2.3. DTIN (Bảng 15, R@10 / R@50)

| Mô hình | cartoon | origami | toy | sculpture | Trung bình |
|---|---|---|---|---|---|
| MagicLens-B (CLIP) | 49.4 / 67.0 | 13.8 / 26.3 | 25.8 / 43.4 | 24.3 / 41.3 | 28.3 / 44.5 |
| MagicLens-L (CLIP) | 62.6 / 72.2 | 21.5 / 33.4 | 43.8 / 58.4 | 38.0 / 54.2 | 41.5 / 54.5 |

Truy vấn là ảnh chụp thật kèm chỉ dẫn `"find this object in {miền}"` ([data_utils.py:253](../data_utils.py#L253)).

## 3. Kết quả tái hiện

Bảng tái hiện nằm ở mục 2.1 của [ket_qua.md](ket_qua.md). Tệp đó cũng có kết quả đầy đủ của bài báo và các thí nghiệm cải tiến của dự án. Hiện mới có FIQ dress (Base): 21.8 / 43.4, so với 21.5 / 41.3 của bài báo.

Khi chạy xong một benchmark, điền số vào bảng đó. Lệch dưới khoảng 1 điểm là bình thường (xem mục 8).

## 4. Môi trường trên Colab

> **Cách nhanh nhất:** dùng các notebook có sẵn trong [notebooks/](../notebooks/). Mỗi notebook làm tự động mọi bước trong mục 4–7 cho một benchmark, kể cả bản sửa RAM của CIRCO. Các mục dưới đây giải thích từng bước, dùng khi muốn chạy tay hoặc khi notebook báo lỗi.

Chọn **Runtime → Change runtime type → A100**. Các khối bash dưới đây chạy trong ô Colab có dòng đầu là `%%bash`. Mỗi ô là một shell mới, nên khối nào cũng `cd` lại từ đầu.

Kiểm tra phần cứng:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
free -g        # RAM hệ thống; DTIN cần khoảng 16 GB
df -h /content # dung lượng đĩa; CIRCO cần khoảng 40 GB lúc giải nén
```

Cài đặt theo README gốc. Tổ hợp này đã chạy được với Python 3.12:

```bash
cd /content
git clone https://github.com/google-deepmind/magiclens.git
git clone https://github.com/google-research/scenic.git
pip install -q ./scenic
pip install -q -r scenic/scenic/projects/baselines/clip/requirements.txt
pip install -q -U "jax[cuda12]"
```

Sau đó chọn **Runtime → Restart session** rồi kiểm tra:

```python
import jax; print(jax.__version__, jax.devices())   # phải thấy CudaDevice
```

Nếu import flax hoặc scenic báo lỗi phiên bản, cài bộ phiên bản đã từng chạy được: `pip install "jax[cuda12]==0.11.1" flax==0.12.9`.

Tải checkpoint:

```python
from google.colab import auth; auth.authenticate_user()
```

```bash
cd /content/magiclens && gsutil -m cp -R gs://gresearch/magiclens/models ./
ls -lh models   # magic_lens_clip_base.pkl, magic_lens_clip_large.pkl
```

Nếu gsutil báo lỗi quyền, tải từ thư mục Google Drive có link trong [README](../README.md). Nên chép thư mục `models` vào Drive một lần để các phiên sau khỏi tải lại.

**Hai lưu ý chung:**
- Mọi lệnh `inference.py` phải chạy từ thư mục gốc của repo, vì mã dùng đường dẫn tương đối `./data/...`.
- Tokenizer tải tệp từ vựng CLIP từ GitHub ở mỗi lần chạy, nên runtime cần có mạng.

Đổi `base` thành `large` trong các lệnh dưới đây để chạy bản Large: `--model_size large --model_path ./models/magic_lens_clip_large.pkl`.

## 5. FashionIQ

**Dữ liệu.** Đường dẫn trong [data/README.md](../data/README.md) trỏ tới [issue #18 của repo fashion-iq](https://github.com/XiaoxiaoGuo/fashion-iq/issues/18). Issue này chia sẻ một tệp `fashion-iq.7z` (978 MB) trên [Yandex Disk](https://disk.yandex.com/d/Z2E54WCwvrQA3A), tải được không cần tài khoản. Bên trong có đúng ba tệp `images.tar.gz`, `captions-….zip`, `image_splits-….zip`, và có đủ ảnh cho tập val của cả ba danh mục. Tải và giải nén trên Colab:

```bash
cd /content/magiclens && mkdir -p data/fiq && cd data/fiq
URL=$(curl -s "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.com/d/Z2E54WCwvrQA3A" \
      | python3 -c "import json, sys; print(json.load(sys.stdin)['href'])")
wget -q --show-progress -O fashion-iq.7z "$URL"
7z x -y fashion-iq.7z && rm fashion-iq.7z     # nếu chưa có lệnh 7z: pip install py7zr && python3 -m py7zr x fashion-iq.7z
tar -xzf images.tar.gz && unzip -q captions-*.zip && unzip -q image_splits-*.zip
rm images.tar.gz captions-*.zip image_splits-*.zip
```

Mã gốc (bản clone trên Colab) sẽ dừng với lỗi ở ảnh thiếu đầu tiên, nên kiểm tra đủ ảnh trước (ô Python). Bản `data_utils.py` trong repo này thì tự bỏ ảnh và truy vấn bị thiếu, và in số lượng còn lại.

```python
import json, os; os.chdir("/content/magiclens")
for t in ["dress", "shirt", "toptee"]:
    split = json.load(open(f"data/fiq/image_splits/split.{t}.val.json"))
    caps = json.load(open(f"data/fiq/captions/cap.{t}.val.json"))
    need = set(split) | {q["candidate"] for q in caps} | {q["target"] for q in caps}
    miss = [i for i in need if not os.path.exists(f"data/fiq/images/{i}.png")]
    print(t, len(caps), "truy vấn,", len(split), "ảnh trong kho, thiếu", len(miss))
```

Kết quả đúng phải khớp số ở mục 2.1 và không thiếu ảnh nào. Nếu tải ảnh từ đường dẫn Amazon gốc thì sẽ thiếu khoảng 4–5% ảnh, khi đó số liệu không khớp hoàn toàn với bài báo.

**Chạy:**

```bash
cd /content/magiclens
for t in dress shirt toptee; do
  python inference.py --model_size base --model_path ./models/magic_lens_clip_base.pkl --dataset fiq-$t
done
```

Mỗi lần chạy in ra một dòng `Recalls:  {10: …, 50: …}`. Cột trung bình là trung bình cộng của ba danh mục.

## 6. DTIN

**Cách tính điểm.** Có 10 000 truy vấn là ảnh chụp thật lấy từ tập val của ImageNet. Kho tìm kiếm có 16 983 ảnh: 6 983 ảnh ImageNet-R thuộc 4 miền, cộng với chính 10 000 ảnh truy vấn đóng vai ảnh gây nhiễu. Một truy vấn được tính là đúng nếu top-K có ảnh cùng lớp trong đúng miền được yêu cầu. Nếu mô hình trả về chính ảnh chụp gốc thì vẫn tính là sai.

**Dữ liệu:**

```bash
cd /content/magiclens && mkdir -p data/dtin && cd data/dtin
BASE=https://raw.githubusercontent.com/google-research/composed_image_retrieval/refs/heads/main/data/imgnet
wget -q $BASE/imgnet_real_query.txt $BASE/imgnet_targets.txt
wget -q https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar && tar -xf imagenet-r.tar && rm imagenet-r.tar
```

Tập val của ImageNet (`ILSVRC2012_img_val.tar`, 6.4 GB): [data/README.md](../data/README.md) tải thẳng bằng `wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar`. Nếu lệnh này không được, đăng ký tài khoản trên image-net.org, tải tệp về rồi đưa lên Drive. Tệp `extract_ILSVRC.sh` mà [data/README.md](../data/README.md) nhắc tới không có trong repo. Thay vào đó, `valprep.sh` làm đúng phần cần thiết: chia ảnh val vào các thư mục theo mã lớp.

```bash
cd /content/magiclens/data/dtin && mkdir -p real
tar -xf /content/drive/MyDrive/ILSVRC2012_img_val.tar -C real
cd real && wget -q https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh && bash valprep.sh && rm valprep.sh
```

Kiểm tra đủ ảnh:

```python
import os; os.chdir("/content/magiclens")
for f in ["imgnet_real_query.txt", "imgnet_targets.txt"]:
    paths = [l.split()[0].split("/", 1)[1] for l in open(f"data/dtin/{f}")]
    print(f, len(paths), "dòng, thiếu", sum(not os.path.exists("data/dtin/" + p) for p in paths))
# Kết quả đúng: 10000 và 16983 dòng, thiếu 0
```

**Chạy:**

```bash
cd /content/magiclens
for d in cartoon origami toy sculpture; do
  python inference.py --model_size base --model_path ./models/magic_lens_clip_base.pkl --dataset dtin-$d
done
```

- Mỗi lần chạy nạp lại 26 983 ảnh (khoảng 16 GB RAM) và mã hóa lại toàn bộ kho. Kho giống hệt nhau ở cả 4 miền, nên việc này lặp lại 4 lần cho cùng một kết quả, nhưng A100 vẫn chạy được.
- Danh sách top-50 được ghi ra `output/dtin-<miền>_base/`.
- Hàm `build_dtin_dataset` có ghi chú `# TODO test` ([data_utils.py:239](../data_utils.py#L239)). Nếu kết quả lệch nhiều so với mục 2.3, hãy kiểm tra hàm này trước.

## 7. CIRCO

**Dữ liệu (khoảng 19 GB).** Bộ này lớn hơn dung lượng Drive miễn phí (15 GB), nên mỗi phiên hãy tải lại:

```bash
cd /content/magiclens && mkdir -p data/circo && cd data/circo
git clone -q https://github.com/miccunifi/CIRCO.git && mv CIRCO/annotations ./ && rm -rf CIRCO
mkdir -p COCO2017_unlabeled && cd COCO2017_unlabeled
wget -q http://images.cocodataset.org/zips/unlabeled2017.zip && unzip -q unlabeled2017.zip && rm unlabeled2017.zip
wget -q http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip && unzip -q image_info_unlabeled2017.zip && rm image_info_unlabeled2017.zip
ls unlabeled2017 | wc -l   # phải là 123403
```

**Bắt buộc sửa mã để không hết RAM.** Mã gốc giữ cả 123 403 ảnh trong RAM, mỗi ảnh là một mảng float32 224 × 224 × 3 (khoảng 0.6 MB), tổng cộng khoảng 74 GB. Gần như chắc chắn runtime sẽ bị dừng vì hết bộ nhớ.

Cách sửa là chỉ lưu đường dẫn, rồi đọc ảnh theo từng lô. Hàm tiền xử lý và mô hình giữ nguyên nên kết quả không đổi. Bản sửa này **chưa được áp dụng** trong repo. Cần sửa hai chỗ:

Trong `build_circo_dataset`, sửa `process_index_example` ([data_utils.py:201](../data_utils.py#L201)) thành:

```python
    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, image_id2name(index_img_id))
        return IndexExample(iid=index_img_id, iimage=img_path, itokens=null_tokens)  # chỉ lưu đường dẫn
```

Trong vòng lặp mã hóa kho ảnh ([inference.py:109](../inference.py#L109)):

```python
from concurrent.futures import ThreadPoolExecutor   # thêm vào đầu tệp
from data_utils import process_img                   # thêm vào đầu tệp
...
  pool = ThreadPoolExecutor()
  for i in tqdm(range(0, len(eval_dataset.index_examples), args.batch_size)):
    batch = eval_dataset.index_examples[i : i + args.batch_size]
    iids = [i.iid for i in batch]
    imgs = [x.iimage for x in batch]
    if isinstance(imgs[0], str):  # CIRCO: đọc ảnh theo lô
      imgs = list(pool.map(lambda p: process_img(p, 224), imgs))
    iimages = jnp.concatenate(imgs, axis=0)
    # các dòng còn lại giữ nguyên
```

**Chạy và nộp kết quả:**

```bash
cd /content/magiclens
python inference.py --model_size base --model_path ./models/magic_lens_clip_base.pkl --dataset circo
# Kết quả ghi ra output/circo_base/circo_results.json (800 truy vấn × top-50)
```

Nộp tệp JSON này lên [máy chủ CIRCO](https://circo.micc.unifi.it/), kết quả được gửi về qua email. Cũng có thể nộp lên [máy chủ thay thế trên Codabench](https://www.codabench.org/competitions/10055/), nhưng phải nén tệp thành zip. Tệp do repo xuất ra đã đúng định dạng yêu cầu: khóa là id truy vấn, giá trị là danh sách 50 id ảnh.

**Kiểm tra tại máy trước khi nộp (tùy chọn).** Tập val có nhãn công khai gồm 220 truy vấn. Đổi `test.json` thành `val.json` ở [data_utils.py:190](../data_utils.py#L190), chạy với `--output ./output_val`, rồi chấm điểm:

```python
import json, numpy as np
gt = {q["id"]: set(q["gt_img_ids"]) for q in json.load(open("data/circo/annotations/val.json"))}
res = json.load(open("output_val/circo_base/circo_results.json"))
for K in (5, 10, 25, 50):
    aps = []
    for qid, g in gt.items():
        hits = np.array([r in g for r in res[str(qid)][:K]], float)
        prec = np.cumsum(hits) / np.arange(1, K + 1)
        aps.append((prec * hits).sum() / min(K, len(g)))
    print(f"mAP@{K}: {100 * np.mean(aps):.1f}")
```

Bài báo không có số trên tập val để so. Bước này chỉ để chắc pipeline chạy đúng: điểm phải cùng cỡ với số trên test, không gần 0. Nhớ đổi lại `test.json` trước khi chạy bản để nộp.

## 8. Đọc kết quả và lỗi thường gặp

**Lệch bao nhiêu là bình thường.** Dưới khoảng 1 điểm là chấp nhận được, do checkpoint chuyển đổi và khác biệt số học nhỏ giữa GPU và CPU. Nếu lệch từ 2 điểm trở lên, kiểm tra theo thứ tự:
1. Dữ liệu có thiếu ảnh không.
2. `--model_size` có khớp với tệp `.pkl` không.
3. Mẫu câu truy vấn trong `data_utils.py` có bị sửa không.

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `FileNotFoundError: ./data/...` | Không chạy từ thư mục gốc của repo, hoặc thiếu ảnh | `cd` về thư mục gốc; chạy lại đoạn kiểm tra ảnh |
| Session bị dừng khi đang nạp ảnh | Hết RAM (CIRCO chưa sửa mã) | Áp dụng bản sửa ở mục 7 |
| `RESOURCE_EXHAUSTED` khi chạy mô hình | Batch 1024 quá lớn, thường gặp với bản Large | Thêm `--batch_size 256` |
| `jax.devices()` chỉ thấy CPU | Đang dùng jax bản CPU, hoặc chưa restart | Cài lại `jax[cuda12]`, rồi Restart session |
| Lỗi khi nạp checkpoint | `--model_size` không khớp với tệp `.pkl` | `base` đi với `magic_lens_clip_base.pkl`, `large` với `magic_lens_clip_large.pkl` |
