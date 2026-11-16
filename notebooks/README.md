# Notebook tái hiện benchmark trên Google Colab

Ba notebook dưới đây chạy lại các benchmark của bài báo MagicLens bằng mã gốc của repo, trên Google Colab với GPU A100. Mỗi notebook tự làm hết các bước: cài thư viện, tải checkpoint, chuẩn bị dữ liệu, chạy `inference.py`, so với số của bài báo và lưu kết quả lên Google Drive.

| Notebook | Benchmark | Bạn cần chuẩn bị | Kết quả |
|---|---|---|---|
| [01_fiq.ipynb](01_fiq.ipynb) | FashionIQ: dress, shirt, toptee | Không cần, tự tải khoảng 1 GB | R@10, R@50, so với Bảng 12 |
| [02_dtin.ipynb](02_dtin.ipynb) | DTIN: cartoon, origami, toy, sculpture | `ILSVRC2012_img_val.tar` trên Drive, nếu notebook không tự tải được | R@10, R@50, so với Bảng 15 |
| [03_circo.ipynb](03_circo.ipynb) | CIRCO | Không cần, tự tải khoảng 19 GB | mAP trên tập val; tệp top-50 của tập test để nộp lên máy chủ CIRCO |

Số liệu của bài báo và giải thích từng benchmark nằm trong [docs/tai_hien_benchmark.md](../docs/tai_hien_benchmark.md).

## Mở notebook trên Colab

Chọn một trong hai cách:

- **Tải tệp lên:** vào [colab.research.google.com](https://colab.research.google.com), chọn **File → Upload notebook**, rồi chọn tệp `.ipynb` trong thư mục này.
- **Mở từ GitHub** (chỉ được khi repo để công khai): chọn **File → Open notebook → GitHub**, dán đường dẫn repo rồi chọn notebook. Hoặc mở thẳng các đường dẫn:
  - https://colab.research.google.com/github/LDH1401/Image_Retrieval/blob/main/notebooks/01_fiq.ipynb
  - https://colab.research.google.com/github/LDH1401/Image_Retrieval/blob/main/notebooks/02_dtin.ipynb
  - https://colab.research.google.com/github/LDH1401/Image_Retrieval/blob/main/notebooks/03_circo.ipynb

Sau khi mở, vào **Runtime → Change runtime type**, chọn **A100 GPU**, rồi chạy từng ô từ trên xuống bằng **Shift + Enter**, hoặc chọn **Runtime → Run all**.

## Dữ liệu trên Google Drive

Các notebook dùng chung thư mục `MyDrive/magiclens/` trên Drive:

```
MyDrive/magiclens/
├── data/
│   ├── fiq/                         ← 01_fiq tự tải và lưu tệp vào đây ở lần chạy đầu
│   │   └── fashion-iq.7z
│   └── dtin/                        ← đặt tệp val của ImageNet vào đây nếu 02_dtin không tự tải được
│       └── ILSVRC2012_img_val.tar
└── ket_qua/                         ← notebook tự tạo: log, bảng so sánh, tệp top-50
    ├── fiq/base/
    ├── dtin/base/
    └── circo/base/
```

- **FashionIQ:** notebook tự tải `fashion-iq.7z` (978 MB, không cần tài khoản) từ [Yandex Disk](https://disk.yandex.com/d/Z2E54WCwvrQA3A). Đây là đường dẫn được chia sẻ trong [issue #18 của repo fashion-iq](https://github.com/XiaoxiaoGuo/fashion-iq/issues/18). Bản này có đủ ảnh cho tập val của cả ba danh mục (đã kiểm tra: 2017, 2038, 1961 truy vấn và 3817, 6346, 5373 ảnh, không thiếu ảnh nào). Notebook lưu một bản lên Drive để các phiên sau khỏi tải lại. Nếu đã có dữ liệu khác, đặt vào `fiq/` (các tệp `.7z`, `.zip`, `.tar.gz` đều được). Nếu bộ đó thiếu ảnh, notebook bỏ ảnh và truy vấn bị thiếu rồi vẫn chạy, đồng thời ghi rõ là kết quả không so trực tiếp được với bài báo.
- **ImageNet val:** nếu Drive chưa có `ILSVRC2012_img_val.tar` (6.4 GB), notebook thử tải thẳng từ image-net.org giống lệnh trong [data/README.md](../data/README.md). Nếu không được, đăng ký tài khoản trên [image-net.org](https://image-net.org), tải tệp về rồi đưa lên Drive. Notebook chỉ lấy đúng 10 000 ảnh cần dùng.
- **Không cần đưa lên Drive:** checkpoint MagicLens, ImageNet-R, nhãn và ảnh CIRCO. Các thứ này được tải công khai ở mỗi phiên.

## Lưu ý

- **Mỗi phiên là một máy mới.** Mỗi lần mở lại notebook là một máy Colab mới, nên phải chạy lại từ ô đầu tiên. Trong cùng một phiên, ô nào đã làm xong sẽ tự bỏ qua.
- **Chạy bản Large:** ở ô 1 đổi `MODEL_SIZE = "large"`, chạy lại ô 1, rồi chạy tiếp từ ô 6 trở xuống.
- **CIRCO:** notebook sửa bản sao mã trên máy Colab để đọc ảnh theo lô, vì mã gốc giữ cả 123 403 ảnh trong RAM (khoảng 74 GB). Repo không bị sửa.
- **Tiết kiệm compute unit:** A100 tốn compute unit theo thời gian kết nối, kể cả khi không chạy gì. Xong việc hãy chọn **Runtime → Disconnect and delete runtime**.
- **Mức độ đã kiểm tra:** mọi ô (trừ ô cài thư viện) đã được chạy thử trên CPU với dữ liệu thu nhỏ. Bản sửa của CIRCO cho ra cùng top-50 với mã gốc. Các notebook chưa được chạy với dữ liệu đầy đủ trên A100, nên thời gian chạy thực tế và việc cài đặt trên Colab cần được xác nhận ở lần chạy đầu tiên.
