"""Cac module thuc nghiem (experimental) cua OncoVision.

Cac thanh phan trong day CHUA duoc tich hop vao pipeline suy luan chinh
(`medical/pipeline.py`). Chung duoc giu lai cho muc dich nghien cuu va thu
nghiem, khong dam bao san sang cho production.

Noi dung:
- `multitask`: mo hinh multi-task (ung thu + modality + vung giai phau +
  chat luong anh + su hien dien khoi u). Can dataset da gan nhan da nhiem vu.
- `self_supervised`: pretrain DINOv2. Yeu cau tai trong so tu mang
  (torch.hub / HuggingFace) nen chi chay khi ONCOVISION_ALLOW_WEIGHT_DOWNLOAD=1.

Muon dung: import truc tiep tu `experimental.multitask` hoac
`experimental.self_supervised`.
"""
