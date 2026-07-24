# Mustard Greens crop–weed dataset

Ảnh ruộng chụp bằng điện thoại, gốc 3072×3072, resize về 1296×1296. Phân đoạn 3 lớp:
nền, cây trồng (crop), cỏ dại (weed).

## Cấu trúc (giống carrot_dataset — cắm thẳng vào code huấn luyện)

```
datasets/mustard_greens/
  images/train/      80 ảnh   NNN_image.png
  images/test/       20 ảnh
  annotations/train/ 80 nhãn  NNN_annotation.png (đen=nền, xanh=crop, đỏ=weed) + .yaml
  annotations/test/  20 nhãn
  masks/             100      NNN_mask.png  (mask thực vật, 255=cây)
  split.csv                   ghi lại ảnh nào thuộc tập nào
```

## Chia tập

100 ảnh, chia **80 train / 20 test** theo tỉ lệ 80:20 ở mức ảnh, xáo trộn cố định
(`seed=42`) một lần duy nhất để tái lập. Cả hai tập đều có đủ crop và weed.

## Phân bố điểm ảnh theo lớp

| Tập | Số ảnh | Nền | Cây trồng | Cỏ dại |
|-----|-------:|-----|-----------|--------|
| train | 80  | 94.7M (70.5%) | 25.3M (18.9%) | 14.3M (10.7%) |
| test  | 20  | 21.8M (65.0%) |  8.0M (23.7%) |  3.8M (11.4%) |
| **tổng** | **100** | **116.5M (69.4%)** | **33.3M (19.8%)** | **18.1M (10.8%)** |

