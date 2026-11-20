# Mẫu ảnh demo OncoVision

Thư mục này chứa ảnh mẫu để test demo OncoVision. Do file ảnh bị gitignore, bạn cần tự chuẩn bị ảnh y khoa thực tế.

## Cấu trúc khuyến nghị

```
data/samples/
├── brain/              # Ảnh MRI não
│   ├── sample1.jpg
│   └── sample2.jpg
├── breast/             # Ảnh siêu âm vú
│   ├── sample1.jpg
│   └── sample2.jpg
├── cervical/           # Ảnh colposcopy
│   ├── sample1.jpg
│   └── sample2.jpg
├── colorectal/         # Ảnh colonoscopy
│   ├── sample1.jpg
│   └── sample2.jpg
├── kidney/             # Ảnh CT thận
│   ├── sample1.jpg
│   └── sample2.jpg
├── liver/              # Ảnh CT gan
│   ├── sample1.jpg
│   └── sample2.jpg
├── lung/               # Ảnh CT/X-ray phổi
│   ├── sample1.jpg
│   └── sample2.jpg
├── pancreas/           # Ảnh CT tụy
│   ├── sample1.jpg
│   └── sample2.jpg
├── prostate/           # Ảnh MRI tuyến tiền liệt
│   ├── sample1.jpg
│   └── sample2.jpg
├── stomach/            # Ảnh nội soi dạ dày
│   ├── sample1.jpg
│   └── sample2.jpg
└── thyroid/            # Ảnh siêu âm tuyến giáp
    ├── sample1.jpg
    └── sample2.jpg
```

## Nguồn ảnh mẫu miễn phí

| Ung thư | Dataset | Link |
|---|---|---|
| U não | BraTS | https://www.medicaldecathlon.com/ |
| U vú | BUSI | https://www.kaggle.com/datasets/aryashmehta/breast-ultrasound-images-dataset |
| U cổ tử cung | Hereditary | https://hereditary.cancer.gov/ |
| U đại trực tràng | Kvasir | https://www.kvasir.com/ |
| U thận | MSD Kidney | https://medicaldecathlon.com/ |
| U gan | MSD Liver | https://medicaldecathlon.com/ |
| U phổi | LIDC-IDRI | https://lidc.idri.org/ |
| U tụy | MSD Pancreas | https://medicaldecathlon.com/ |
| U tuyến tiền liệt | Prostate-MRI | https://www.kaggle.com/datasets/arnabchaki/prostate-mri |
| U dạ dày | Kvasir | https://www.kvasir.com/ |
| U tuyến giáp | DDTI | https://www.kaggle.com/datasets/andrewmvd/thyroid-ultrasound-images-dataset |

## Định dạng hỗ trợ

- **Ảnh 2D:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.dcm` (DICOM)
- **Ảnh 3D:** `.nii`, `.nii.gz` (NIfTI) — cần Volume 3D model

## Lưu ý

- Ảnh mẫu chỉ dùng cho **demo nội bộ**, không dùng cho chẩn đoán y khoa.
- Đảm bảo ảnh tuân thủ quy định bảo mật bệnh nhân (de-identified).
- Mỗi ảnh nên từ 224×224 đến 1024×1024 pixel.
