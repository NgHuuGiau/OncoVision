# Backup Directory — Dọn dẹp model 2026-11-20

Thư mục này chứa các file model đã được di chuyển ra khỏi `pretrained/cancers/` để tối ưu hóa cho demo.

## Cấu trúc backup

### volume_3d_broken/ (983.9 MB)
Tất cả file `.pt2` Volume 3D — **không hoạt động** do PyTorch version mismatch.
Hệ thống tự động chạy chế độ 2D khi không có Volume 3D.

| File | Nguồn gốc | Kích thước |
|---|---|---|
| brain_volume_model.pt2 | brain/mri/ | 16.5 MB |
| breast_volume_model.pt2 | breast/ | 2.1 MB |
| cervical_volume_model.pt2 | cervical/colposcopy/ | 2.1 MB |
| colonoscopy_volume.pt2 | colorectal/colonoscopy/ | 94.4 MB |
| colorectal_volume_model.pt2 | colorectal/endoscopy/ | 2.1 MB |
| colposcopy_volume.pt2 | cervical/colposcopy/ | 94.4 MB |
| ct_volume.pt2 | kidney/ct/ | 94.4 MB |
| endoscopy_volume.pt2 | stomach/endoscopy/ | 94.4 MB |
| kidney_volume_model.pt2 | kidney/ct/ | 2.1 MB |
| liver_ct_volume.pt2 | liver/ct/ | 94.4 MB |
| liver_volume_model.pt2 | liver/ct/ | 2.1 MB |
| lung_ct_volume.pt2 | lung/ct/ | 94.4 MB |
| lung_volume_model.pt2 | lung/ct/ | 2.1 MB |
| mri_volume.pt2 | prostate/mri/ | 94.4 MB |
| pancreas_ct_volume.pt2 | pancreas/ct/ | 94.4 MB |
| pancreas_volume_model.pt2 | pancreas/ct/ | 2.1 MB |
| prostate_volume_model.pt2 | prostate/mri/ | 2.1 MB |
| stomach_volume_model.pt2 | stomach/endoscopy/ | 2.4 MB |
| thyroid_thyroid_volume_model.pt2 | thyroid/ | 2.1 MB |
| thyroid_volume_model.pt2 | thyroid/ultrasound/ | 2.4 MB |
| ultrasound_volume.pt2 | breast/ultrasound/ | 94.4 MB |
| volume.pt2 | breast/mammogram/ | 94.4 MB |

### redundant_pth/ (~1.1 GB)
File `.pth` backup — không được hệ thống sử dụng (hệ thống chỉ load `.pt`).

| File | Nguồn gốc | Kích thước |
|---|---|---|
| breast_resnet50_approach1.pth | breast/ | 90 MB |
| breast_resnet50_approach2.pth | breast/ | 90 MB |
| cervical_model.pth | cervical/pap/ | 16.8 MB |
| colorectal_best.pth | colorectal/endoscopy/ | 269.7 MB |
| kidney_ann.pth | kidney/ct/ | 0 MB |
| kidney_cancer_classifier_final_best_acc.pth | kidney/ct/ | 90 MB |
| kidney_classifier.pth | kidney/ct/ | 90 MB |
| liver_tumor_seg.pth | liver/ct/ | 86.9 MB |
| liver_tumor_seg2.pth | liver/ct/ | 86.9 MB |
| liver_unet.pth | liver/ct/ | 18.4 MB |
| liver_unet2.pth | liver/ct/ | 18.4 MB |
| best_metric_model.pth | liver/ | 18.4 MB |
| pancreas_medvit.pth | pancreas/ct/ | 49.2 MB |
| pancreas_medvit_final.pth | pancreas/ct/ | 49.2 MB |
| best_customcnn_model.pth | prostate/mri/ | 134.5 MB |
| gastric_expert_fold0.pth | stomach/endoscopy/ | 232.2 MB |
| gastric_specialist_adi.pth | stomach/endoscopy/ | 77.8 MB |

### unused_safetensors/ (~800 MB)
File `.safetensors` — không được pipeline OncoVision load (chỉ dùng trong `import_foreign_model.py`).

| File | Nguồn gốc | Kích thước |
|---|---|---|
| brain_glioma_segformer.safetensors | brain/ | 14.2 MB |
| brain_meningioma_segformer.safetensors | brain/ | 14.2 MB |
| brain_model.safetensors | brain/ | 26.9 MB |
| breast_checkpoint.safetensors | breast/ | 171.1 MB |
| breast_segformer.safetensors | breast/ | 14.2 MB |
| cervical_colposcopy_model.safetensors | cervical/colposcopy/ | 101.7 MB |
| liver_ct_liver_swin.safetensors | liver/ct/ | 105.2 MB |
| liver_ct_model.safetensors | liver/ct/ | 89.9 MB |
| lung_ct_lung_yolo_tiny.safetensors | lung/ct/ | 24.7 MB |
| prostate_mri_prostate_foundation.safetensors | prostate/mri/ | 327.3 MB |
| thyroid_ultrasound_thyroid_efficientnet.safetensors | thyroid/ultrasound/ | 15.5 MB |
| thyroid_ultrasound_thyroid_resnet50.safetensors | thyroid/ultrasound/ | 89.9 MB |

## Tổng kết

| Thành phần | Kích thước |
|---|---|
| Active models (sau dọn dẹp) | **1.93 GB** |
| Backup (có thể xóa nếu không cần) | **3.32 GB** |
| Tổng thư mục models/ | **5.86 GB** |

## Khôi phục file

Nếu cần khôi phục file từ backup, chỉ cần copy lại vào vị trí gốc trong `pretrained/cancers/`.
