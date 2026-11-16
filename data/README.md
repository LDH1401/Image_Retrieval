## CIRCO
```
mkdir circo
cd circo
```

Trong thư mục `magic_lens/data/circo`

### Tải nhãn (annotations)

```
git clone https://github.com/miccunifi/CIRCO.git
mv ./CIRCO/annotations ./
# rm -rf ./CIRCO (không dùng đến)
```

### Tải ảnh
```
mkdir COCO2017_unlabeled
# tải ảnh
wget http://images.cocodataset.org/zips/unlabeled2017.zip
unzip unlabeled2017.zip -d COCO2017_unlabeled
# tải thông tin ảnh
wget http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
unzip image_info_unlabeled2017.zip -d COCO2017_unlabeled
```

Cấu trúc thư mục cần có như sau (không hiển thị repo CIRCO):
```
circo
└─── annotations
        | test.json
        | val.json
└─── COCO2017_unlabeled
    └─── annotations
        | image_info_unlabeled2017.json

    └─── unlabeled2017
        | 000000243611.jpg
        | 000000535009.jpg
        | 000000097553.jpg
        | ...
```


## FashionIQ (FIQ)
```
mkdir fiq
cd fiq
```

### Tải FIQ
Tải bộ dữ liệu theo hướng dẫn tại [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq/issues/18)

Giải nén tương ứng các tệp `images.tar.gz`, `image_splits-20220326T130551Z-001.zip` và `captions-20220326T130604Z-001.zip`.

Cấu trúc thư mục cần có như sau (không hiển thị các tệp train/test.json):
```
├── README.md
├── captions
│   ├── cap.dress.val.json
│   ├── cap.shirt.val.json
│   └── cap.toptee.val.json
├── image_splits
│   ├── split.dress.val.json
│   ├── split.shirt.val.json
│   └── split.toptee.val.json
├── images
│   ├── B00HINZY58.png
│   ├── B00I0XXRJU.png
│   ├── B00J2UZLNU.png
│   └── B00J66MDMC.png
    ...
```



## Domain-Transfer ImageNet
```
mkdir dtin
cd dtin

# lấy nhãn và truy vấn từ Pic2Word
wget https://raw.githubusercontent.com/google-research/composed_image_retrieval/refs/heads/main/data/imgnet/imgnet_real_query.txt
wget https://raw.githubusercontent.com/google-research/composed_image_retrieval/refs/heads/main/data/imgnet/imgnet_targets.txt
```

### Tải DTIN
Tải bộ dữ liệu ImageNet-R theo hướng dẫn tại [ImageNet R](https://github.com/hendrycks/imagenet-r?tab=readme-ov-file)
Tải tập validation của ImageNet theo hướng dẫn tại [ImageNet Real](https://github.com/hendrycks/imagenet-real?tab=readme-ov-file)

Giải nén tương ứng các tệp `imagenet-r.tar` và `imagenet-real.tar`.
```
wget https://people.eecs.berkeley.edu/\~hendrycks/imagenet-r.tar
tar -xf imagenet-r.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate

tar -xf ILSVRC2012_img_val.tar
sh ./extract_ILSVRC.sh
```

Cấu trúc thư mục cần có như sau (không hiển thị các tệp train/test.json):

```bash
├── dtin
    ├── imagenet-r ## các thư mục ảnh imagenet-r sau khi giải nén. Thư mục này phải chứa các thư mục con.
            └──n01443537
                  .
                  . 
    ├── imgnet_real_query.txt
    ├── imgnet_targets.txt
    └── real ## các thư mục ảnh validation của imagenet. Thư mục này phải chứa các thư mục con.
        └──n01440764
                  .
                  . 
```

