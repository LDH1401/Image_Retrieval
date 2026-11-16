# MagicLens

Kho mã này chứa phần cài đặt của MagicLens. Mã nguồn sử dụng Jax và Flax.
Lưu ý rằng bản cài đặt hiện tại chưa hỗ trợ huấn luyện.
Xem [trang web](https://open-vision-language.github.io/MagicLens/) để tham khảo các ví dụ về bộ dữ liệu.

## Tóm tắt

Chúng tôi giới thiệu MagicLens, một họ mô hình truy xuất ảnh tự giám sát (self-supervised) hỗ trợ
chỉ dẫn mở (open-ended instructions). Luận điểm cốt lõi của MagicLens là chỉ dẫn
bằng văn bản có thể giúp truy xuất những ảnh có
quan hệ phong phú hơn, vượt ra ngoài sự tương đồng về thị giác. MagicLens được xây dựng dựa trên một
nhận định mới then chốt: các cặp ảnh xuất hiện tự nhiên
trên cùng một trang web chứa rất nhiều loại quan hệ ngầm (ví dụ: góc nhìn bên trong của), và chúng ta
có thể biến những quan hệ ngầm đó thành tường minh bằng cách tổng hợp chỉ dẫn thông qua các mô hình đa phương thức lớn (LMM) và mô hình ngôn ngữ lớn (LLM).
Được huấn luyện trên 36.7 triệu bộ ba (ảnh truy vấn, chỉ dẫn, ảnh đích) với các quan hệ ngữ nghĩa phong phú
khai thác từ web, MagicLens đạt kết quả tương đương hoặc tốt hơn trên tám benchmark thuộc
nhiều tác vụ truy xuất ảnh khác nhau so với các phương pháp tốt nhất trước đó (SOTA). Đáng chú ý, mô hình vượt SOTA trước đó trong khi có kích thước
nhỏ hơn 50 lần trên nhiều benchmark. Các
phân tích bổ sung do con người thực hiện trên một kho 1.4 triệu ảnh chưa từng thấy
càng cho thấy sự đa dạng của các ý định tìm kiếm
mà MagicLens hỗ trợ.
![Ảnh giới thiệu](https://open-vision-language.github.io/MagicLens/static/images/magiclens_overview.png)

## Cài đặt
```
conda create --name magic_lens python=3.9
conda activate magic_lens
git clone https://github.com/google-research/scenic.git
cd scenic
pip install .
pip install -r scenic/projects/baselines/clip/requirements.txt
# có thể bạn cần cài phiên bản jax cho GPU tương ứng theo hướng dẫn tại https://jax.readthedocs.io/en/latest/installation.html
# ví dụ:
# # Cài đặt cho CUDA 12
# Lưu ý: các bản wheel chỉ có cho Linux.
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # Cài đặt cho CUDA 11
# Lưu ý: các bản wheel chỉ có cho Linux.
# pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Tải mô hình
Tải mô hình bằng lệnh:
```
cd .. # ở thư mục chính `magiclens`
# có thể bạn cần chạy `gcloud auth login` để có quyền truy cập, tài khoản gmail nào cũng được.
gsutil cp -R gs://gresearch/magiclens/models ./
```

HOẶC qua [google drive](https://drive.google.com/drive/folders/1MXszMqIIh-yV7cYxWUxP7uHs9gfuTT3u)

### Chuẩn bị dữ liệu
Vui lòng làm theo hướng dẫn trong thư mục của từng bộ dữ liệu ở `./data`. Hiện tại chúng tôi đã kiểm thử thành công FIQ, CIRCO và DTIN:

## Suy luận
```
python inference.py \
--model_size large \
--model_path ./models/magic_lens_clip_large.pkl \
--dataset circo

```

Do trọng số đã được chuyển đổi, kết quả có thể chênh lệch đôi chút:

Trên `CIRCO`
| Mô hình | map@5 | map@10 | map@25 | map@50 |
|----------|----------|----------|----------|----------|
| SOTA trước đó | 26.8 | 27.6 | 30.0 | 31.0 |
| Base (gốc) | 23.1 | 23.8 | 25.8 | 26.7 |
| Base (đã chuyển đổi) | 22.3 | 23.2 | 25.0 | 26.0 |
| Large (gốc) | 29.6 | 30.8 | 33.4 | 34.4 |
| Large (đã chuyển đổi) | 29.5 | 30.8 | 33.2 | 34.3 |

## Tài liệu và thí nghiệm của dự án

Phần này không có trong repo gốc.

- [docs/kien_truc_pipeline.md](docs/kien_truc_pipeline.md): kiến trúc và pipeline của MagicLens.
- [docs/ket_qua.md](docs/ket_qua.md): kết quả của bài báo và các kết quả dự án đã có.
- [docs/tai_hien_benchmark.md](docs/tai_hien_benchmark.md) và [notebooks/](notebooks/): tái hiện các benchmark của bài báo trên Google Colab.
- [docs/huong_dan_vlm_rerank.md](docs/huong_dan_vlm_rerank.md): rerank top-K của MagicLens bằng VLM (Qwen2-VL) trên FashionIQ.
- [docs/huong1_tinh_chinh_rerank.md](docs/huong1_tinh_chinh_rerank.md): Hướng 1, ghép MagicLens đã tinh chỉnh với rerank bằng VLM.

## Trích dẫn công trình này

Thêm thông tin trích dẫn tại đây, thường là một đoạn BibTeX có thể sao chép và dán:

```latex
@inproceedings{zhang2024magiclens,
  title = 	 {{M}agic{L}ens: Self-Supervised Image Retrieval with Open-Ended Instructions},
  author =       {Zhang, Kai and Luan, Yi and Hu, Hexiang and Lee, Kenton and Qiao, Siyuan and Chen, Wenhu and Su, Yu and Chang, Ming-Wei},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {59403--59420},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v235/zhang24an.html}
}

```

## Giấy phép và tuyên bố miễn trừ trách nhiệm

*Đây là bản dịch tham khảo; bản tiếng Anh gốc của các giấy phép là bản có giá trị pháp lý.*

Bản quyền 2024 thuộc DeepMind Technologies Limited

Toàn bộ phần mềm được cấp phép theo Giấy phép Apache, Phiên bản 2.0 (Apache 2.0);
bạn không được sử dụng tệp này trừ khi tuân thủ giấy phép Apache 2.0.
Bạn có thể xem bản sao của giấy phép Apache 2.0 tại:
https://www.apache.org/licenses/LICENSE-2.0

Tất cả các tài liệu khác được cấp phép theo Giấy phép Quốc tế Creative Commons
Ghi công 4.0 (CC-BY). Bạn có thể xem bản sao của giấy phép CC-BY tại:
https://creativecommons.org/licenses/by/4.0/legalcode

Trừ khi luật hiện hành yêu cầu hoặc có thỏa thuận bằng văn bản, toàn bộ phần mềm và
tài liệu được phân phối tại đây theo giấy phép Apache 2.0 hoặc CC-BY đều được
phân phối trên cơ sở "NGUYÊN TRẠNG" ("AS IS"), KHÔNG CÓ BẤT KỲ BẢO ĐẢM HAY ĐIỀU KIỆN NÀO,
dù rõ ràng hay ngụ ý. Xem các giấy phép để biết quy định cụ thể về quyền hạn
và giới hạn theo từng giấy phép.

Đây không phải là sản phẩm chính thức của Google.
