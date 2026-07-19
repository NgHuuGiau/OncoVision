# Experimental Modules

Cac module trong thu muc nay **chua duoc tich hop** vao pipeline suy luan chinh
(`medical/pipeline.py`). Chung la code nghien cuu/thu nghiem, khong dam bao san
sang cho production va co the thay doi hoac bi go bo.

## Noi dung

| Module | Mo ta | Yeu cau |
|---|---|---|
| `multitask/` | Mo hinh multi-task (cancer + modality + anatomical + quality + tumor presence) | Dataset da gan nhan da nhiem vu |
| `self_supervised/` | Pretrain DINOv2 cho anh y khoa | Tai trong so tu mang: dat `ONCOVISION_ALLOW_WEIGHT_DOWNLOAD=1` |

## Vi sao tach rieng

- Tranh gay hieu nham ve muc do hoan thien cua he thong: cac module da tich hop
  (segmentation ROI, uncertainty MC Dropout) nam trong `medical/`, con phan chua
  tich hop nam o day.
- Chinh sach mac dinh cua he thong la **khong tai trong so tu mang tru YOLO**.
  `self_supervised` can mang nen khong the la mot phan cua luong runtime mac dinh.

## Cach dung

```python
from experimental.multitask import MultiTaskMedicalModel
from experimental.self_supervised import DINOv2Pretrainer  # can cho phep tai mang
```
